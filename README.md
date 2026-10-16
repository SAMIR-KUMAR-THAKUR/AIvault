# AIvault
AIVault is a real-time dashboard for monitoring AI usage and cost. It tracks tokens, spend, latency and errors across OpenAI, Anthropic, Gemini and Llama. A Python SDK sends the data to a FastAPI backend, which stores it in SQLite and streams it live to the dashboard over WebSocket. It also has budget alerts and CSV/JSON export.
