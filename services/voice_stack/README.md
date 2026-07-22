# Xeon voice stack

Always-hot English speech services for ClawBody:

- `POST /v1/audio/transcriptions`: faster-whisper `distil-large-v3`
- `POST /v1/audio/speech`: Chatterbox Turbo cloned from a configured voice reference
- `GET /health`: loaded-model readiness

The service intentionally exposes an OpenAI-shaped API contract so ClawBody can switch providers without changing its conversation loop. It binds to loopback by default; use the Reachy LAN address only after firewalling it to the robot subnet.

```bash
uv sync
ASMO_VOICE_REFERENCE=/path/to/asmo.wav uv run uvicorn app:app --host 127.0.0.1 --port 8890
```
