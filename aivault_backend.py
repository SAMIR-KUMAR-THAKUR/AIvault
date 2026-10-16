
import asyncio
import io
import csv
import json
import os
import sys
import time
import sqlite3
import threading
from collections import deque, defaultdict
from datetime import datetime, date
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from pydantic import BaseModel

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GPU_AVAILABLE = False
GPU_COUNT     = 0
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_COUNT     = pynvml.nvmlDeviceGetCount()
    GPU_AVAILABLE = True
except Exception:
    pass

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
WS_INTERVAL       = 2.0   # seconds between WebSocket broadcasts
DB_PATH           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aivault.db")

COST_PER_1K: Dict[str, float] = {
    # OpenAI
    "gpt-4.5-turbo":                  0.030,
    "gpt-4o":                         0.010,
    "gpt-4o-mini":                    0.0015,
    "gpt-4-turbo":                    0.020,
    "gpt-3.5-turbo":                  0.001,
    "o1":                             0.030,
    "o3-mini":                        0.002,
    # Anthropic
    "claude-3-5-sonnet-20241022":     0.015,
    "claude-3-5-sonnet":              0.015,
    "claude-3-5-haiku-20241022":      0.004,
    "claude-3-5-haiku":               0.004,
    "claude-3-haiku-20240307":        0.0025,
    "claude-3-opus-20240229":         0.075,
    "claude-3-opus":                  0.075,
    # Meta / Open-source
    "llama-3.3-70b":                  0.0001,
    "llama-3.1-8b":                   0.00003,
    "llama-3.1-70b":                  0.0001,
    "mistral-large":                  0.004,
    # Google Gemini
    "gemini-1.5-pro":                 0.007,
    "gemini-1.5-flash":               0.00035,
    "gemini-2.0-flash":               0.0003,
    "gemini-2.5-pro":                 0.0125,
    # Embeddings
    "text-embedding-3-large":         0.0001,
    "text-embedding-3-small":         0.00002,
}

HEALTH_ENDPOINTS = [
    {"name": "OpenAI · gpt-4.5-turbo",       "url": "https://api.openai.com/v1/models"},
    {"name": "Anthropic · claude-3-5-sonnet", "url": "https://api.anthropic.com/v1/models"},
    {"name": "vLLM · llama-3.3-70b (local)", "url": "http://localhost:8080/health"},
    {"name": "Pinecone · prod-cluster",       "url": "https://api.pinecone.io/indexes"},
]

_START_TIME = time.time()



class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS token_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    is_error INTEGER NOT NULL DEFAULT 0,
                    tag TEXT DEFAULT '',
                    user_id TEXT DEFAULT ''
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON token_events(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_model ON token_events(model)")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS budget_settings (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL
                )
            """)
            cur.execute("INSERT OR IGNORE INTO budget_settings (key, value) VALUES ('daily_budget', 50.0)")
            cur.execute("INSERT OR IGNORE INTO budget_settings (key, value) VALUES ('monthly_budget', 1500.0)")
            conn.commit()
            conn.close()

    def insert_events(self, events: List[dict]):
        if not events:
            return
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.executemany("""
                INSERT INTO token_events 
                (timestamp, model, prompt_tokens, completion_tokens, total_tokens, cost, latency_ms, is_error, tag, user_id)
                VALUES (:timestamp, :model, :prompt_tokens, :completion_tokens, :total_tokens, :cost, :latency_ms, :is_error, :tag, :user_id)
            """, events)
            conn.commit()
            conn.close()

    def get_recent_events(self, limit: int = 50, model: Optional[str] = None, is_error: Optional[bool] = None) -> List[dict]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            query = "SELECT * FROM token_events"
            conditions = []
            params = []
            if model:
                conditions.append("model LIKE ?")
                params.append(f"%{model}%")
            if is_error is not None:
                conditions.append("is_error = ?")
                params.append(1 if is_error else 0)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows

    def load_today_stats(self) -> dict:
        today_str = date.today().isoformat()
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT model,
                       SUM(prompt_tokens) as ptoks,
                       SUM(completion_tokens) as ctoks,
                       SUM(total_tokens) as ttoks,
                       SUM(cost) as total_cost,
                       COUNT(*) as req_count,
                       SUM(CASE WHEN is_error=1 THEN 1 ELSE 0 END) as err_count,
                       AVG(latency_ms) as avg_lat
                FROM token_events
                WHERE timestamp >= ?
                GROUP BY model
            """, (today_str + "T00:00:00",))
            model_stats = cur.fetchall()

            cur.execute("""
                SELECT SUM(total_tokens), SUM(cost), COUNT(*)
                FROM token_events
                WHERE timestamp >= ?
            """, (today_str + "T00:00:00",))
            tot_row = cur.fetchone()

            conn.close()
            return {
                "models": [dict(r) for r in model_stats],
                "total_tokens": tot_row[0] or 0,
                "total_cost": tot_row[1] or 0.0,
                "total_requests": tot_row[2] or 0,
            }

    def get_budget(self) -> dict:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT key, value FROM budget_settings")
            rows = dict(cur.fetchall())
            conn.close()
            return {
                "daily_budget": float(rows.get("daily_budget", 50.0)),
                "monthly_budget": float(rows.get("monthly_budget", 1500.0)),
            }

    def set_budget(self, daily: Optional[float] = None, monthly: Optional[float] = None):
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            if daily is not None:
                cur.execute("INSERT OR REPLACE INTO budget_settings (key, value) VALUES ('daily_budget', ?)", (daily,))
            if monthly is not None:
                cur.execute("INSERT OR REPLACE INTO budget_settings (key, value) VALUES ('monthly_budget', ?)", (monthly,))
            conn.commit()
            conn.close()

    def clear_all(self):
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM token_events")
            conn.commit()
            conn.close()


db = DatabaseManager()


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN STORE (Thread-Safe In-Memory Aggregator + Async DB Flush)
# ─────────────────────────────────────────────────────────────────────────────
class TokenStore:
    def __init__(self):
        self._lock          = threading.RLock()
        self.reset_day      = date.today()
        self.models         = defaultdict(lambda: {
            "tokens": 0, "cost": 0.0, "requests": 0, "errors": 0,
            "latencies": deque(maxlen=200),
        })
        self.total_tokens   = 0
        self.total_cost     = 0.0
        self.total_requests = 0
        self.tps_window     = deque(maxlen=500)       # (timestamp, tokens)
        self.sparkline      = deque([0.0] * 60, maxlen=60)
        self._cost_this_sec = 0.0
        self.req_volume     = deque([0] * 20, maxlen=20)
        self.recent_logs    = deque(maxlen=25)        # circular buffer of recent logs

        # Asynchronous DB writer queue
        self._db_queue      = deque()
        self._db_thread     = threading.Thread(target=self._db_worker, daemon=True)
        self._db_thread.start()

        # Load persisted today stats from SQLite
        self._bootstrap_from_db()

    def _bootstrap_from_db(self):
        try:
            stats = db.load_today_stats()
            with self._lock:
                self.total_tokens   = stats["total_tokens"]
                self.total_cost     = stats["total_cost"]
                self.total_requests = stats["total_requests"]
                for row in stats["models"]:
                    m = self.models[row["model"]]
                    m["tokens"]   = row["ttoks"]
                    m["cost"]     = row["total_cost"]
                    m["requests"] = row["req_count"]
                    m["errors"]   = row["err_count"]
                    if row["avg_lat"]:
                        m["latencies"].append(row["avg_lat"])
            # Populate initial recent logs
            recents = db.get_recent_events(limit=25)
            for r in reversed(recents):
                self.recent_logs.append(r)
        except Exception as e:
            print(f"[AIVAULT] Bootstrap error: {e}")

    def _db_worker(self):
        """Worker thread that flushes queued events to SQLite."""
        while True:
            batch = []
            with self._lock:
                while self._db_queue and len(batch) < 100:
                    batch.append(self._db_queue.popleft())
            if batch:
                try:
                    db.insert_events(batch)
                except Exception as e:
                    print(f"[AIVAULT] DB flush error: {e}")
            time.sleep(0.5)

    def record(self, model: str, prompt_tok: int, comp_tok: int,
               lat_ms: float = 0.0, is_error: bool = False,
               tag: str = "", user_id: str = ""):
        total = prompt_tok + comp_tok
        cost_unit = COST_PER_1K.get(model)
        if cost_unit is None:
            # Fallback estimation based on model family
            low = model.lower()
            if "flash" in low or "mini" in low or "haiku" in low or "8b" in low:
                cost_unit = 0.0005
            elif "claude" in low or "gpt-4" in low or "o1" in low or "opus" in low:
                cost_unit = 0.015
            else:
                cost_unit = 0.002
        cost = total / 1000.0 * cost_unit

        ts = datetime.utcnow().isoformat() + "Z"
        event = {
            "timestamp":         ts,
            "model":             model,
            "prompt_tokens":     prompt_tok,
            "completion_tokens": comp_tok,
            "total_tokens":      total,
            "cost":              round(cost, 6),
            "latency_ms":        round(lat_ms, 1),
            "is_error":          1 if is_error else 0,
            "tag":               tag or "",
            "user_id":           user_id or "",
        }

        with self._lock:
            m = self.models[model]
            m["tokens"]   += total
            m["cost"]     += cost
            m["requests"] += 1
            if is_error:
                m["errors"] += 1
            if lat_ms > 0:
                m["latencies"].append(lat_ms)
            self.total_tokens   += total
            self.total_cost     += cost
            self.total_requests += 1
            self._cost_this_sec += cost
            self.tps_window.append((time.time(), total))
            self.req_volume.append(total)
            self.recent_logs.append(event)
            self._db_queue.append(event)

    def reset(self):
        """Reset counters and clear database."""
        with self._lock:
            self.models.clear()
            self.total_tokens   = 0
            self.total_cost     = 0.0
            self.total_requests = 0
            self._cost_this_sec = 0.0
            self.tps_window.clear()
            self.sparkline      = deque([0.0] * 60, maxlen=60)
            self.req_volume     = deque([0] * 20, maxlen=20)
            self.recent_logs.clear()
            self._db_queue.clear()
            self.reset_day      = date.today()
        db.clear_all()

    def tick_sparkline(self):
        with self._lock:
            self.sparkline.append(self._cost_this_sec)
            self._cost_this_sec = 0.0

    def tps(self, window: float = 5.0) -> float:
        cutoff = time.time() - window
        with self._lock:
            tokens = [t for ts, t in self.tps_window if ts >= cutoff]
        return sum(tokens) / window if tokens else 0.0

    def p95(self) -> float:
        with self._lock:
            lats = []
            for m in self.models.values():
                lats.extend(m["latencies"])
        if not lats:
            return 0.0
        lats.sort()
        return lats[int(len(lats) * 0.95)]

    def snapshot(self) -> dict:
        with self._lock:
            tps_val = self.tps()
            uptime  = max(time.time() - _START_TIME, 1)
            spark   = list(self.sparkline)
            max_s   = max(spark) if any(spark) else 1
            spark_pct = [round(v / max_s * 100, 1) if max_s else 5 for v in spark]

            models_out = {}
            for name, m in self.models.items():
                lats = list(m["latencies"])
                p95  = lats[int(len(lats) * 0.95)] if lats else 0.0
                share = m["tokens"] / max(self.total_tokens, 1)
                models_out[name] = {
                    "tokens":   m["tokens"],
                    "cost":     round(m["cost"], 4),
                    "requests": m["requests"],
                    "errors":   m["errors"],
                    "p95_ms":   round(p95, 1),
                    "share":    round(share * 100, 1),
                }

            logs = list(self.recent_logs)

            return {
                "total_tokens":   self.total_tokens,
                "total_cost":     round(self.total_cost, 4),
                "total_requests": self.total_requests,
                "tps":            round(tps_val, 1),
                "cost_per_sec":   round(self.total_cost / uptime, 6),
                "p95_ms":         round(self.p95(), 1),
                "spark_pct":      spark_pct,
                "req_volume":     list(self.req_volume),
                "models":         models_out,
                "recent_logs":    logs[-10:],
            }


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH MONITOR
# ─────────────────────────────────────────────────────────────────────────────
class HealthMonitor:
    def __init__(self):
        self.results: Dict[str, dict] = {
            ep["name"]: {"name": ep["name"], "status": "CHECKING",
                         "latency_ms": 0, "uptime_pct": 100.0}
            for ep in HEALTH_ENDPOINTS
        }
        self._hits   = defaultdict(int)
        self._total  = defaultdict(int)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import urllib.request
        time.sleep(3)
        while True:
            for ep in HEALTH_ENDPOINTS:
                name = ep["name"]
                t0   = time.perf_counter()
                ok   = False
                try:
                    req = urllib.request.Request(ep["url"], headers={"User-Agent": "AIVault/1.0"})
                    urllib.request.urlopen(req, timeout=4)
                    ok = True
                except Exception:
                    pass
                lat_ms = (time.perf_counter() - t0) * 1000

                self._total[name] += 1
                if ok:
                    self._hits[name] += 1
                uptime = self._hits[name] / self._total[name] * 100

                if "4.5" in name:
                    status = "DEPRECATING" if ok else "DOWN"
                elif ok:
                    status = "HEALTHY" if lat_ms < 1500 else "DEGRADED"
                else:
                    status = "DOWN"

                self.results[name] = {
                    "name":       name,
                    "status":     status,
                    "latency_ms": round(lat_ms, 1),
                    "uptime_pct": round(uptime, 2),
                }
            time.sleep(30)

    def snapshot(self) -> list:
        return list(self.results.values())


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM MONITOR
# ─────────────────────────────────────────────────────────────────────────────
class SystemMonitor:
    def __init__(self):
        self._net_prev = psutil.net_io_counters()
        self._net_time = time.time()

    def snapshot(self) -> dict:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        net     = psutil.net_io_counters()
        dt      = max(time.time() - self._net_time, 0.001)
        bps_in  = (net.bytes_recv - self._net_prev.bytes_recv) / dt
        bps_out = (net.bytes_sent - self._net_prev.bytes_sent) / dt
        self._net_prev = net
        self._net_time = time.time()

        def fmt(bps):
            if bps >= 1e9: return f"{bps/1e9:.1f} GB/s"
            if bps >= 1e6: return f"{bps/1e6:.1f} MB/s"
            return f"{bps/1e3:.0f} KB/s"

        gpus = []
        if GPU_AVAILABLE:
            for i in range(GPU_COUNT):
                try:
                    h   = pynvml.nvmlDeviceGetHandleByIndex(i)
                    u   = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m   = pynvml.nvmlDeviceGetMemoryInfo(h)
                    nm  = pynvml.nvmlDeviceGetName(h)
                    nm  = nm if isinstance(nm, str) else nm.decode()
                    gpus.append({
                        "index": i, "name": nm,
                        "util_pct":    u.gpu,
                        "mem_used_gb": round(m.used  / 1e9, 2),
                        "mem_total_gb":round(m.total / 1e9, 2),
                    })
                except Exception:
                    pass

        return {
            "cpu_pct":     round(cpu, 1),
            "ram_pct":     round(ram.percent, 1),
            "net_in":      fmt(bps_in),
            "net_out":     fmt(bps_out),
            "gpu_count":   GPU_COUNT,
            "gpus":        gpus,
            "gpu_util":    (gpus[0]["util_pct"] if gpus else 0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────
token_store = TokenStore()
health_mon  = HealthMonitor()
sys_mon     = SystemMonitor()
ws_clients: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(_sparkline_ticker())
    asyncio.create_task(_ws_broadcaster())
    print("=" * 60)
    print("  🚀 AIVAULT Backend — Real-Time AI FinOps Engine")
    print("  WebSocket     : ws://localhost:8000/ws")
    print("  REST Metrics  : http://localhost:8000/api/metrics")
    print("  Activity Logs : http://localhost:8000/api/logs")
    print("  Budget Alerts : http://localhost:8000/api/budget")
    print("  Data Export   : http://localhost:8000/api/export")
    print("  Database      : " + DB_PATH)
    print(f"  Hardware      : {GPU_COUNT} GPU(s)" if GPU_AVAILABLE else "  Hardware      : CPU Mode")
    print("=" * 60)
    yield


app = FastAPI(title="AIVAULT Intelligence Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC BROADCASTER & PAYLOAD BUILDER
# ─────────────────────────────────────────────────────────────────────────────
async def _sparkline_ticker():
    while True:
        await asyncio.sleep(1)
        token_store.tick_sparkline()


async def _ws_broadcaster():
    while True:
        await asyncio.sleep(WS_INTERVAL)
        if not ws_clients:
            continue
        payload = _build_payload()
        msg     = json.dumps(payload)
        dead    = []
        for ws in ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.remove(ws)


def _build_payload() -> dict:
    tok  = token_store.snapshot()
    sys  = sys_mon.snapshot()
    hlth = health_mon.snapshot()
    bgt  = db.get_budget()

    def fmt_tok(n):
        if n >= 1e9: return f"{n/1e9:.2f}B"
        if n >= 1e6: return f"{n/1e6:.1f}M"
        if n >= 1e3: return f"{n/1e3:.0f}k"
        return str(n)

    # Smart matching for model blocks
    models_dict = tok["models"]

    def find_model_key(patterns):
        for k in models_dict:
            for p in patterns:
                if p.lower() in k.lower():
                    return k
        return None

    def model_block(patterns, default_name):
        key = find_model_key(patterns)
        m = models_dict.get(key, {}) if key else {}
        t  = m.get("tokens", 0)
        c  = m.get("cost", 0.0)
        sh = m.get("share", 0)
        p95 = m.get("p95_ms", 0)
        return {
            "name":   key or default_name,
            "tokens": fmt_tok(t),
            "cost":   f"${c:.2f} today",
            "tps":    f"{tok['tps'] * sh/100:.0f} tok/s",
            "bar":    min(100, sh if t > 0 else 0),
            "p95":    f"{p95:.0f}ms",
        }

    # Budget calculation
    spent_today  = tok["total_cost"]
    daily_limit  = bgt["daily_budget"]
    daily_pct    = round((spent_today / max(daily_limit, 0.01)) * 100, 1)
    if daily_pct >= 100:
        budget_status = "CRITICAL"
    elif daily_pct >= 80:
        budget_status = "WARNING"
    else:
        budget_status = "OK"

    return {
        "total_tokens":   fmt_tok(tok["total_tokens"]),
        "running_cost":   f"${tok['total_cost']:.2f}",
        "tps_rate":       f"{tok['tps']:,.0f} tok/s",
        "cost_rate":      f"${tok['cost_per_sec']:.4f}/s",
        "p95_latency_ms": f"{tok['p95_ms']:.0f}ms",
        "cost_per_min":   f"${tok['cost_per_sec']*60:.3f}/min",
        "total_requests": tok["total_requests"],
        "cost_sparkline": tok["spark_pct"],
        "req_volume":     tok["req_volume"],

        # Model cards for UI
        "gpt45":  model_block(["gpt-4.5", "gpt-4o", "gpt-4"], "gpt-4.5-turbo"),
        "claude": model_block(["claude-3-5", "claude-3", "sonnet"], "claude-3-5-sonnet-20241022"),
        "llama":  model_block(["llama-3.3", "llama-3.1", "llama"], "llama-3.3-70b"),
        "embed":  model_block(["embed", "text-embedding"], "text-embedding-3-large"),

        # Budget info
        "budget": {
            "daily_budget":  daily_limit,
            "spent_today":   spent_today,
            "daily_pct":     daily_pct,
            "status":        budget_status,
        },

        # System info
        "gpu_util":       f"{sys['gpu_util']}%",
        "net_in":         sys["net_in"],
        "net_out":        sys["net_out"],
        "cpu_pct":        sys["cpu_pct"],
        "ram_pct":        sys["ram_pct"],
        "health":         hlth,
        "recent_logs":    tok["recent_logs"],
        "all_models":     models_dict,
        "ts":             datetime.utcnow().isoformat() + "Z",
        "backend_live":   True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REST API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class TrackRequest(BaseModel):
    model:             str
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    latency_ms:        float = 0.0
    is_error:          bool  = False
    tag:               str   = ""
    user_id:           str   = ""


class BatchTrackRequest(BaseModel):
    events: List[TrackRequest]


class BudgetUpdate(BaseModel):
    daily_budget:   Optional[float] = None
    monthly_budget: Optional[float] = None


@app.get("/api/metrics")
async def get_metrics():
    return {
        "tokens": token_store.snapshot(),
        "system": sys_mon.snapshot(),
        "health": health_mon.snapshot(),
        "budget": db.get_budget(),
        "uptime": round(time.time() - _START_TIME, 1)
    }


@app.post("/api/track")
async def track_endpoint(req: TrackRequest):
    """Log a single token event."""
    token_store.record(
        req.model, req.prompt_tokens, req.completion_tokens,
        req.latency_ms, req.is_error, req.tag, req.user_id
    )
    return {"ok": True}


@app.post("/api/track/batch")
async def track_batch_endpoint(req: BatchTrackRequest):
    """High-throughput batch ingestion endpoint."""
    for ev in req.events:
        token_store.record(
            ev.model, ev.prompt_tokens, ev.completion_tokens,
            ev.latency_ms, ev.is_error, ev.tag, ev.user_id
        )
    return {"ok": True, "count": len(req.events)}


@app.get("/api/logs")
async def get_logs(limit: int = Query(50, ge=1, le=500),
                   model: Optional[str] = None,
                   is_error: Optional[bool] = None):
    """Retrieve recent request event logs with optional filters."""
    logs = db.get_recent_events(limit=limit, model=model, is_error=is_error)
    return {"logs": logs, "count": len(logs)}


@app.get("/api/budget")
async def get_budget():
    """Retrieve budget and spend alert thresholds."""
    bgt = db.get_budget()
    tok = token_store.snapshot()
    today_cost = tok["total_cost"]
    daily_limit = bgt["daily_budget"]
    monthly_limit = bgt["monthly_budget"]
    pct = round((today_cost / max(daily_limit, 0.01)) * 100, 1)
    status = "CRITICAL" if pct >= 100 else ("WARNING" if pct >= 80 else "OK")
    return {
        "daily_budget":   daily_limit,
        "monthly_budget": monthly_limit,
        "spent_today":    today_cost,
        "daily_pct":      pct,
        "alert_status":   status,
    }


@app.post("/api/budget")
async def update_budget(req: BudgetUpdate):
    """Update daily or monthly budget limits."""
    db.set_budget(daily=req.daily_budget, monthly=req.monthly_budget)
    return {"ok": True, "budget": db.get_budget()}


@app.get("/api/export")
async def export_data(format: str = Query("csv", pattern="^(csv|json)$")):
    """Export tracked telemetry logs as CSV or JSON for FinOps analysis."""
    events = db.get_recent_events(limit=5000)
    if format == "json":
        return JSONResponse(events)

    # Return CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id", "timestamp", "model", "prompt_tokens", "completion_tokens",
        "total_tokens", "cost", "latency_ms", "is_error", "tag", "user_id"
    ])
    writer.writeheader()
    for ev in events:
        writer.writerow(ev)
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aivault_token_export.csv"}
    )


@app.post("/api/reset")
async def reset_stats():
    """Reset all token counters, logs, and database records."""
    token_store.reset()
    return {"ok": True, "message": "All stats and database records reset to zero."}


@app.get("/api/models")
async def get_models():
    return token_store.snapshot()["models"]


@app.get("/api/system")
async def get_system():
    return sys_mon.snapshot()


@app.get("/api/health")
async def get_health():
    return health_mon.snapshot()


# ─────────────────────────────────────────────────────────────────────────────
# PROXIES: ZERO-CODE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

# OpenAI Proxy
@app.api_route("/proxy/openai/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_openai(path: str, request: Request):
    api_key = request.headers.get("authorization", "").replace("Bearer ", "").strip() or OPENAI_API_KEY
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not set and no Authorization header provided")
    import httpx as _httpx
    body = await request.body()
    hdrs = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    hdrs["Authorization"] = f"Bearer {api_key}"
    t0 = time.perf_counter()
    try:
        async with _httpx.AsyncClient(timeout=120) as c:
            r = await c.request(request.method, f"https://api.openai.com/v1/{path}", headers=hdrs, content=body)
        lat = (time.perf_counter() - t0) * 1000
        try:
            d = r.json()
            u = d.get("usage", {})
            token_store.record(d.get("model", "unknown"),
                               u.get("prompt_tokens", 0),
                               u.get("completion_tokens", 0),
                               lat, r.status_code >= 400, tag="openai_proxy")
        except Exception:
            pass
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
    except Exception as e:
        raise HTTPException(502, f"Proxy error: {str(e)}")


# Anthropic Proxy
@app.api_route("/proxy/anthropic/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_anthropic(path: str, request: Request):
    api_key = request.headers.get("x-api-key", "").strip() or ANTHROPIC_API_KEY
    if not api_key:
        raise HTTPException(503, "ANTHROPIC_API_KEY not set and no x-api-key header provided")
    import httpx as _httpx
    body = await request.body()
    hdrs = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    hdrs["x-api-key"] = api_key
    hdrs["anthropic-version"] = request.headers.get("anthropic-version", "2023-06-01")
    t0 = time.perf_counter()
    try:
        async with _httpx.AsyncClient(timeout=120) as c:
            r = await c.request(request.method, f"https://api.anthropic.com/v1/{path}", headers=hdrs, content=body)
        lat = (time.perf_counter() - t0) * 1000
        try:
            d = r.json()
            u = d.get("usage", {})
            token_store.record(d.get("model", "unknown"),
                               u.get("input_tokens", 0),
                               u.get("output_tokens", 0),
                               lat, r.status_code >= 400, tag="anthropic_proxy")
        except Exception:
            pass
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
    except Exception as e:
        raise HTTPException(502, f"Anthropic proxy error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    print(f"[WS] Client connected ({len(ws_clients)} total)")
    try:
        await ws.send_text(json.dumps(_build_payload()))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
        print(f"[WS] Client disconnected ({len(ws_clients)} remaining)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
