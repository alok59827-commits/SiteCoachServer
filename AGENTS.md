# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Site Coach is a single-file FastAPI backend (`main.py`) that accepts construction-site audio recordings, transcribes them via Deepgram, and returns AI coaching feedback via Groq LLM. There is no frontend, database, or Docker setup.

### Running the dev server

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server exposes two endpoints:
- `GET /` — health check
- `POST /upload-audio` — accepts multipart form with `file` (audio), `audience`, and `output_language` fields

Interactive API docs are available at `http://localhost:8000/docs`.

### Required environment variables

Both must be set before starting the server for full functionality:
- `DEEPGRAM_API_KEY` — Deepgram speech-to-text API key
- `GROQ_API_KEY` — Groq LLM API key

Without these keys the server starts but `/upload-audio` returns a 401 error from Deepgram.

### Testing notes

- There are no automated tests in this repository.
- No linter or type-checker configuration exists.
- End-to-end testing requires a real audio file and valid API keys.
- `pip install -r requirements.txt` installs to `~/.local`; ensure `~/.local/bin` is on `PATH` for `uvicorn`.
