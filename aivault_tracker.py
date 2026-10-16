
import time
import functools
import threading
import queue
import requests
from typing import Optional, Any, Iterator, AsyncIterator
from contextlib import contextmanager

# ─── Config ──────────────────────────────────────────────────────────────────
AIVAULT_URL   = "http://localhost:8000"   # change if backend runs elsewhere
_TIMEOUT      = 2.0                        # network timeout
_ENABLED      = True                       # set False to disable all tracking
_BATCH_SIZE   = 20                         # flush when queue reaches this size
_FLUSH_PERIOD = 0.3                        # flush every 300ms if queue not empty

# Thread-local storage for context sessions (tags, user_id)
_context = threading.local()


def _get_context():
    tag = getattr(_context, "tag", "")
    user_id = getattr(_context, "user_id", "")
    return tag, user_id


@contextmanager
def session(tag: str = "", user_id: str = ""):
    """
    Context manager to attach metadata (tag, user_id) to all LLM calls within scope.

    Example:
        with aivault_tracker.session(tag="support_agent", user_id="user_123"):
            client.chat.completions.create(...)  # Automatically tagged
    """
    prev_tag = getattr(_context, "tag", "")
    prev_user = getattr(_context, "user_id", "")
    _context.tag = tag or prev_tag
    _context.user_id = user_id or prev_user
    try:
        yield
    finally:
        _context.tag = prev_tag
        _context.user_id = prev_user


# Alias for session
trace = session


# ══════════════════════════════════════════════════════════════════════════════
# ASYNCHRONOUS BATCH QUEUE WORKER
# ══════════════════════════════════════════════════════════════════════════════

class _BatchWorker:
    def __init__(self):
        self._q = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AIVaultWorker")
        self._thread.start()

    def enqueue(self, payload: dict):
        if not _ENABLED:
            return
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            pass  # Drop under extreme backpressure to protect host application

    def _worker_loop(self):
        while not self._stop_event.is_set():
            batch = []
            try:
                # Wait up to _FLUSH_PERIOD for at least one item
                item = self._q.get(timeout=_FLUSH_PERIOD)
                batch.append(item)
                # Drain additional items up to _BATCH_SIZE
                while len(batch) < _BATCH_SIZE:
                    try:
                        batch.append(self._q.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if batch:
                self._send_batch(batch)
                for _ in batch:
                    self._q.task_done()

    def _send_batch(self, batch: list):
        try:
            if len(batch) == 1:
                requests.post(f"{AIVAULT_URL}/api/track", json=batch[0], timeout=_TIMEOUT)
            else:
                requests.post(f"{AIVAULT_URL}/api/track/batch", json={"events": batch}, timeout=_TIMEOUT)
        except Exception:
            pass  # Tracking must never crash or block the host app

    def flush(self, timeout: float = 2.0):
        """Block until all currently queued events are sent."""
        try:
            deadline = time.time() + timeout
            while not self._q.empty() and time.time() < deadline:
                time.sleep(0.05)
        except Exception:
            pass


_worker = _BatchWorker()


def flush(timeout: float = 2.0):
    """Wait for queued telemetry events to be sent to AIVault backend."""
    _worker.flush(timeout)


def track(model: str, prompt_tokens: int = 0, completion_tokens: int = 0,
          latency_ms: float = 0.0, is_error: bool = False,
          tag: str = "", user_id: str = ""):
    """
    Track an LLM token event. Non-blocking; enqueues immediately for batch dispatch.

    Example:
        track("gpt-4o", prompt_tokens=512, completion_tokens=128, latency_ms=680)
    """
    ctx_tag, ctx_user = _get_context()
    _worker.enqueue({
        "model":             model,
        "prompt_tokens":     int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "latency_ms":        float(latency_ms),
        "is_error":          bool(is_error),
        "tag":               tag or ctx_tag or "",
        "user_id":           user_id or ctx_user or "",
    })


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING INTERCEPTORS
# ══════════════════════════════════════════════════════════════════════════════

def _wrap_sync_stream(stream_iter, model_name: str, t0: float, tag: str = "", user_id: str = ""):
    """Wrap a sync stream iterator to extract tokens upon stream end."""
    prompt_tokens = 0
    completion_tokens = 0
    captured_model = model_name
    is_err = False

    try:
        for chunk in stream_iter:
            # Check for usage object in chunk (e.g. OpenAI stream_options)
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
            if hasattr(chunk, "model") and chunk.model:
                captured_model = chunk.model
            # Rough token estimate if usage isn't provided
            choices = getattr(chunk, "choices", None)
            if choices and len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    completion_tokens += max(len(delta.content) // 4, 1)
            yield chunk
    except Exception:
        is_err = True
        raise
    finally:
        lat_ms = (time.perf_counter() - t0) * 1000
        track(
            model=captured_model,
            prompt_tokens=prompt_tokens or 20,
            completion_tokens=completion_tokens or 1,
            latency_ms=lat_ms,
            is_error=is_err,
            tag=tag or "streaming",
            user_id=user_id,
        )


async def _wrap_async_stream(stream_iter, model_name: str, t0: float, tag: str = "", user_id: str = ""):
    """Wrap an async stream iterator to extract tokens upon stream end."""
    prompt_tokens = 0
    completion_tokens = 0
    captured_model = model_name
    is_err = False

    try:
        async for chunk in stream_iter:
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
            if hasattr(chunk, "model") and chunk.model:
                captured_model = chunk.model
            choices = getattr(chunk, "choices", None)
            if choices and len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    completion_tokens += max(len(delta.content) // 4, 1)
            yield chunk
    except Exception:
        is_err = True
        raise
    finally:
        lat_ms = (time.perf_counter() - t0) * 1000
        track(
            model=captured_model,
            prompt_tokens=prompt_tokens or 20,
            completion_tokens=completion_tokens or 1,
            latency_ms=lat_ms,
            is_error=is_err,
            tag=tag or "streaming_async",
            user_id=user_id,
        )


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI SDK PATCHING (Sync & Async, Streaming & Non-Streaming)
# ══════════════════════════════════════════════════════════════════════════════

def patch_openai():
    """Wrap the OpenAI client so every chat/completion call is auto-tracked."""
    try:
        from openai.resources.chat.completions import Completions as ChatCompletions
        from openai.resources.completions import Completions

        _orig_chat = ChatCompletions.create

        @functools.wraps(_orig_chat)
        def _tracked_chat(self, *args, **kwargs):
            t0 = time.perf_counter()
            model_req = kwargs.get("model", "unknown")
            is_stream = kwargs.get("stream", False)
            try:
                resp = _orig_chat(self, *args, **kwargs)
                if is_stream:
                    return _wrap_sync_stream(resp, model_req, t0)
                usage = getattr(resp, "usage", None)
                if usage:
                    track(
                        model=getattr(resp, "model", model_req),
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        is_error=False,
                    )
                return resp
            except Exception:
                track(model=model_req, latency_ms=(time.perf_counter() - t0) * 1000, is_error=True)
                raise

        ChatCompletions.create = _tracked_chat

        _orig_comp = Completions.create

        @functools.wraps(_orig_comp)
        def _tracked_comp(self, *args, **kwargs):
            t0 = time.perf_counter()
            model_req = kwargs.get("model", "unknown")
            try:
                resp = _orig_comp(self, *args, **kwargs)
                usage = getattr(resp, "usage", None)
                if usage:
                    track(
                        model=getattr(resp, "model", model_req),
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                return resp
            except Exception:
                track(model=model_req, latency_ms=(time.perf_counter() - t0) * 1000, is_error=True)
                raise

        Completions.create = _tracked_comp
        print("[AIVAULT] OpenAI SDK (sync) patched successfully")
    except ImportError:
        pass


def patch_openai_async():
    """Wrap AsyncOpenAI client so async chat/completions calls are tracked."""
    try:
        from openai.resources.chat.completions import AsyncCompletions

        _orig_async_chat = AsyncCompletions.create

        @functools.wraps(_orig_async_chat)
        async def _tracked_async_chat(self, *args, **kwargs):
            t0 = time.perf_counter()
            model_req = kwargs.get("model", "unknown")
            is_stream = kwargs.get("stream", False)
            try:
                resp = await _orig_async_chat(self, *args, **kwargs)
                if is_stream:
                    return _wrap_async_stream(resp, model_req, t0)
                usage = getattr(resp, "usage", None)
                if usage:
                    track(
                        model=getattr(resp, "model", model_req),
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        is_error=False,
                    )
                return resp
            except Exception:
                track(model=model_req, latency_ms=(time.perf_counter() - t0) * 1000, is_error=True)
                raise

        AsyncCompletions.create = _tracked_async_chat
        print("[AIVAULT] AsyncOpenAI SDK patched successfully")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC SDK PATCHING
# ══════════════════════════════════════════════════════════════════════════════

def patch_anthropic():
    """Wrap Anthropic messages client so calls are tracked."""
    try:
        import anthropic
        from anthropic.resources.messages import Messages

        _orig = Messages.create

        @functools.wraps(_orig)
        def _tracked(self, *args, **kwargs):
            t0 = time.perf_counter()
            model_req = kwargs.get("model", "claude-unknown")
            is_stream = kwargs.get("stream", False)
            try:
                resp = _orig(self, *args, **kwargs)
                if is_stream:
                    return _wrap_sync_stream(resp, model_req, t0)
                usage = getattr(resp, "usage", None)
                if usage:
                    track(
                        model=getattr(resp, "model", model_req),
                        prompt_tokens=getattr(usage, "input_tokens", 0),
                        completion_tokens=getattr(usage, "output_tokens", 0),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                return resp
            except Exception:
                track(model=model_req, latency_ms=(time.perf_counter() - t0) * 1000, is_error=True)
                raise

        Messages.create = _tracked
        print("[AIVAULT] Anthropic SDK (sync) patched successfully")
    except ImportError:
        pass


def patch_anthropic_async():
    """Wrap AsyncAnthropic client so calls are tracked."""
    try:
        from anthropic.resources.messages import AsyncMessages

        _orig_async = AsyncMessages.create

        @functools.wraps(_orig_async)
        async def _tracked_async_msgs(self, *args, **kwargs):
            t0 = time.perf_counter()
            model_req = kwargs.get("model", "claude-unknown")
            is_stream = kwargs.get("stream", False)
            try:
                resp = await _orig_async(self, *args, **kwargs)
                if is_stream:
                    return _wrap_async_stream(resp, model_req, t0)
                usage = getattr(resp, "usage", None)
                if usage:
                    track(
                        model=getattr(resp, "model", model_req),
                        prompt_tokens=getattr(usage, "input_tokens", 0),
                        completion_tokens=getattr(usage, "output_tokens", 0),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                return resp
            except Exception:
                track(model=model_req, latency_ms=(time.perf_counter() - t0) * 1000, is_error=True)
                raise

        AsyncMessages.create = _tracked_async_msgs
        print("[AIVAULT] AsyncAnthropic SDK patched successfully")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE GEMINI SDK PATCHING
# ══════════════════════════════════════════════════════════════════════════════

def patch_gemini():
    """Wrap Google Generative AI (Gemini) SDK so calls are tracked."""
    try:
        from google.generativeai.generative_models import GenerativeModel

        _orig_generate = GenerativeModel.generate_content

        @functools.wraps(_orig_generate)
        def _tracked_generate(self, *args, **kwargs):
            t0 = time.perf_counter()
            model_name = getattr(self, "model_name", "gemini-unknown").replace("models/", "")
            try:
                resp = _orig_generate(self, *args, **kwargs)
                lat_ms = (time.perf_counter() - t0) * 1000
                usage = getattr(resp, "usage_metadata", None)
                if usage:
                    track(
                        model=model_name,
                        prompt_tokens=getattr(usage, "prompt_token_count", 0),
                        completion_tokens=getattr(usage, "candidates_token_count", 0),
                        latency_ms=lat_ms,
                    )
                else:
                    track(model=model_name, latency_ms=lat_ms)
                return resp
            except Exception:
                track(model=model_name, latency_ms=(time.perf_counter() - t0) * 1000, is_error=True)
                raise

        GenerativeModel.generate_content = _tracked_generate

        try:
            _orig_async_gen = GenerativeModel.generate_content_async

            @functools.wraps(_orig_async_gen)
            async def _tracked_generate_async(self, *args, **kwargs):
                t0 = time.perf_counter()
                model_name = getattr(self, "model_name", "gemini-unknown").replace("models/", "")
                try:
                    resp = await _orig_async_gen(self, *args, **kwargs)
                    lat_ms = (time.perf_counter() - t0) * 1000
                    usage = getattr(resp, "usage_metadata", None)
                    if usage:
                        track(
                            model=model_name,
                            prompt_tokens=getattr(usage, "prompt_token_count", 0),
                            completion_tokens=getattr(usage, "candidates_token_count", 0),
                            latency_ms=lat_ms,
                        )
                    else:
                        track(model=model_name, latency_ms=lat_ms)
                    return resp
                except Exception:
                    track(model=model_name, latency_ms=(time.perf_counter() - t0) * 1000, is_error=True)
                    raise

            GenerativeModel.generate_content_async = _tracked_generate_async
        except AttributeError:
            pass

        print("[AIVAULT] Google Gemini SDK patched successfully")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE TRANSFORMERS PIPELINE PATCHING
# ══════════════════════════════════════════════════════════════════════════════

def patch_huggingface():
    """Wrap HuggingFace pipeline calls for local model tracking."""
    try:
        from transformers import Pipeline
        _orig_call = Pipeline.__call__

        @functools.wraps(_orig_call)
        def _tracked_call(self, inputs, *args, **kwargs):
            t0 = time.perf_counter()
            model_name = getattr(self.model, "name_or_path", "local-hf-model")
            try:
                tokenizer = self.tokenizer
                if isinstance(inputs, str):
                    prompt_tok = len(tokenizer.encode(inputs))
                elif isinstance(inputs, list):
                    prompt_tok = sum(len(tokenizer.encode(i)) for i in inputs if isinstance(i, str))
                else:
                    prompt_tok = len(str(inputs)) // 4
            except Exception:
                prompt_tok = len(str(inputs)) // 4

            result = _orig_call(self, inputs, *args, **kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000

            try:
                out_text = ""
                if isinstance(result, list) and result:
                    first = result[0]
                    if isinstance(first, dict):
                        out_text = first.get("generated_text", "")
                    elif isinstance(first, list) and first:
                        out_text = first[0].get("generated_text", "")
                out_tok = len(self.tokenizer.encode(out_text)) if out_text else 20
            except Exception:
                out_tok = 20

            track(model_name, prompt_tok, out_tok, latency_ms)
            return result

        Pipeline.__call__ = _tracked_call
        print("[AIVAULT] HuggingFace Pipeline patched successfully")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# @track_llm DECORATOR
# ══════════════════════════════════════════════════════════════════════════════

def track_llm(model: str = "unknown", tag: str = ""):
    """Decorator that auto-tracks tokens and latency for any custom LLM function."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                lat_ms = (time.perf_counter() - t0) * 1000
                usage = getattr(result, "usage", None)
                if usage:
                    track(
                        model=getattr(result, "model", model),
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0),
                        latency_ms=lat_ms,
                        tag=tag,
                    )
                else:
                    chars = sum(len(str(a)) for a in args) + sum(len(str(v)) for v in kwargs.values())
                    track(model=model, prompt_tokens=chars // 4, completion_tokens=50, latency_ms=lat_ms, tag=tag)
                return result
            except Exception:
                lat_ms = (time.perf_counter() - t0) * 1000
                track(model=model, latency_ms=lat_ms, is_error=True, tag=tag)
                raise
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# LANGCHAIN CALLBACK HANDLER
# ══════════════════════════════════════════════════════════════════════════════

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    class AIVaultCallback(BaseCallbackHandler):
        def __init__(self, tag: str = "langchain"):
            self._t0: dict = {}
            self.tag = tag

        def on_llm_start(self, serialized, prompts, **kwargs):
            run_id = str(kwargs.get("run_id", id(self)))
            self._t0[run_id] = time.perf_counter()

        def on_llm_end(self, response: LLMResult, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            t0     = self._t0.pop(run_id, time.perf_counter())
            lat_ms = (time.perf_counter() - t0) * 1000

            model_name = "unknown"
            for gen_list in response.generations:
                for gen in gen_list:
                    info = getattr(gen, "generation_info", {}) or {}
                    usage = info.get("usage", {})
                    model_name = info.get("model", model_name)
                    ptoks = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                    ctoks = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                    if ptoks + ctoks > 0:
                        track(model_name, ptoks, ctoks, lat_ms, tag=self.tag)
                        return

            lo = response.llm_output or {}
            usage = lo.get("token_usage", lo.get("usage", {}))
            model_name = lo.get("model_name", model_name)
            ptoks = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            ctoks = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            track(model_name, ptoks, ctoks, lat_ms, tag=self.tag)

        def on_llm_error(self, error, **kwargs):
            run_id = str(kwargs.get("run_id", ""))
            t0     = self._t0.pop(run_id, time.perf_counter())
            lat_ms = (time.perf_counter() - t0) * 1000
            track("unknown", latency_ms=lat_ms, is_error=True, tag=self.tag)

except ImportError:
    class AIVaultCallback:  # type: ignore
        def __init__(self, *a, **kw):
            print("[AIVAULT] langchain-core not installed — AIVaultCallback unavailable")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI / STARLETTE MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    import json as _json

    class AIVaultMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            t0   = time.perf_counter()
            resp = await call_next(request)
            lat  = (time.perf_counter() - t0) * 1000

            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                body_bytes = b""
                async for chunk in resp.body_iterator:
                    body_bytes += chunk

                try:
                    data  = _json.loads(body_bytes)
                    usage = data.get("usage", {})
                    model = data.get("model", "")
                    if model and usage:
                        ptoks = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                        ctoks = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                        if ptoks + ctoks > 0:
                            track(model, ptoks, ctoks, lat, is_error=resp.status_code >= 400, tag="middleware")
                except Exception:
                    pass

                from starlette.responses import Response
                return Response(
                    content=body_bytes,
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                    media_type=resp.media_type,
                )
            return resp

except ImportError:
    class AIVaultMiddleware:  # type: ignore
        def __init__(self, *a, **kw):
            print("[AIVAULT] starlette not installed — AIVaultMiddleware unavailable")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER PATCH ALL
# ══════════════════════════════════════════════════════════════════════════════

def patch_all():
    """
    Call this ONE line at the top of your application to track every AI call:
    Patches OpenAI (sync & async), Anthropic (sync & async),
    Google Gemini (sync & async), and HuggingFace pipelines.
    """
    patch_openai()
    patch_openai_async()
    patch_anthropic()
    patch_anthropic_async()
    patch_gemini()
    patch_huggingface()


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_connection() -> bool:
    """Returns True if the AIVAULT backend is reachable."""
    try:
        r = requests.get(f"{AIVAULT_URL}/api/metrics", timeout=2)
        if r.status_code == 200:
            print(f"[AIVAULT] Backend connected at {AIVAULT_URL}")
            return True
    except Exception:
        pass
    print(f"[AIVAULT] Backend not reachable at {AIVAULT_URL} — run: python aivault_backend.py")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Test Script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("AIVAULT Tracker — Running connection & ingestion test...")
    if check_connection():
        print("Sending sample multi-provider events...")
        track("gpt-4o", prompt_tokens=250, completion_tokens=120, latency_ms=410, tag="chat_demo")
        track("claude-3-5-sonnet-20241022", prompt_tokens=350, completion_tokens=180, latency_ms=530, tag="rag_agent")
        track("llama-3.3-70b", prompt_tokens=400, completion_tokens=90, latency_ms=85, tag="local_vllm")
        track("gemini-1.5-pro", prompt_tokens=300, completion_tokens=150, latency_ms=480, tag="multimodal")

        # Test context session
        with session(tag="checkout_copilot", user_id="user_88"):
            track("gpt-4o-mini", prompt_tokens=150, completion_tokens=40, latency_ms=190)

        # Flush queue before exiting
        flush()
        print("All events sent and flushed successfully! Check dashboard: aivault_v3.html")
    else:
        print("Start the backend first: python aivault_backend.py")
