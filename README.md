# Sahara Pay Voice Agent

Built for the **MLC (Africa) × Intron Agentic Voice AI Challenge**, Deep Learning Indaba 2026.
Category: **Fintech, Telco & Customer Experience**. Code-switching languages: **Akan, Pidgin, and Swahili — English mixed with a user-selected local language.**

## Problem & target users

Mobile money and telco customer-care lines across Africa serve callers who
naturally code-switch between English and a local language mid-sentence —
especially numbers, amounts, and product names ("me pɛ sɛ me send 50 cedis kɔ
Ama"). Speech systems built for clean, monolingual audio break down exactly
where they're needed most: real customer requests. Sahara Pay is a voice
agent that transcribes these mixed requests and **actually completes the
transaction** — check balance, send money, buy an airtime/data bundle, or log
a complaint — rather than just producing a transcript.

## Solution overview

1. **Pick a language** (Akan / Pidgin / Swahili) and **record** a spoken request in the browser.
2. **Transcribe** the same clip through three STT engines concurrently, for a
   fair, reproducible benchmark:
   - **Intron Sahara** (`use_language_asr_input` = `ak` / `pcm` / `sw`) — the language-specialized engine.
   - **Groq Whisper-large-v3** (free tier) — global commercial-grade engine, not tuned for these languages.
   - **Local `faster-whisper`** — open-source/offline baseline, not tuned for these languages.
3. **Act**: the Sahara transcript is sent to an LLM (Groq's `llama-3.3-70b-versatile`,
   free tier, function-calling) that classifies intent (`check_balance` /
   `send_money` / `buy_bundle` / `file_complaint`) and extracts slots, then
   the backend **executes** it against an in-memory mock wallet — this is the
   agentic step, it mutates real state.
4. **Confirm**: a natural-language confirmation is shown, with an optional
   spoken confirmation via Sahara TTS (feature-flagged, degrades to text-only
   if TTS isn't ready in time).

## Why it's free to run (aside from Sahara)

Both the commercial STT baseline and the agent's reasoning LLM run on
[Groq's free tier](https://console.groq.com/keys) (no credit card required)
via the OpenAI-compatible `openai` SDK pointed at Groq's `base_url` — no
separate SDK, no OpenAI cost. The third engine, `faster-whisper`, runs
entirely locally. Only the Sahara calls require the challenge-provided API
key.

## Why Sahara wins the benchmark here

The other two engines are included specifically *because* they weren't tuned
for Akan, Pidgin, or Swahili — the benchmark report should show Sahara
clearly outperforming both on code-switched audio, which is the whole point
of the challenge. See `benchmark/report.md` (generated, not hand-written)
after running the benchmark script below.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in INTRON_API_KEY and GROQ_API_KEY (free)
uvicorn app:app --reload
```

Open http://localhost:8000, pick a language, click **Start Recording**, speak
a request, click **Stop Recording**. The three transcripts appear side by
side with latency, then the agent's action and the updated mock wallet state.

Try:
- "Check my balance"
- "Send 20 cedis to Ama" (mix in Akan/Pidgin/Swahili as you like)
- "Buy me 1GB data"
- "I want to file a complaint about my network"

`faster-whisper` downloads its model weights on first use (a few hundred MB) —
run it once before a live demo so there's no download delay on stage.

## Benchmark report

```bash
python benchmark/run_benchmark.py
```

Drop code-switched clips into `benchmark/samples/ak/`, `benchmark/samples/pcm/`,
or `benchmark/samples/sw/` (optionally with a matching `<clip>.txt` reference
transcript for a rough accuracy score). This reuses the exact same engine code
as the live app and writes `benchmark/report.md`.

## Deploying so teammates can test it (Hugging Face Spaces)

Running locally only works while your laptop is on and reachable, so for a
URL teammates can hit anytime, deploy to Render's free web service tier
(512MB RAM, no credit card required). Note: Hugging Face Spaces was the
original plan here, but its free `cpu-basic` tier turned out to require a
PRO subscription for Docker/Gradio SDKs as of this challenge — static-only
Spaces are free, but this app needs a real backend, so that ruled it out.

1. Push this project to a GitHub repo (you'll do the actual `git push`
   yourself — that's outside what gets automated here):
   ```bash
   git init   # if this folder isn't already a git repo
   git add .
   git commit -m "Sahara Pay Voice Agent"
   git remote add origin https://github.com/<your-username>/sahara-voice-agent.git
   git push -u origin main
   ```
2. Create a free account at [render.com](https://render.com) if you don't
   have one (GitHub sign-in is fastest), then **New → Web Service** → connect
   the repo you just pushed. Render auto-detects the `Dockerfile` at the
   project root — no other config needed.
3. Under **Environment**, add: `INTRON_API_KEY`, `GROQ_API_KEY`, and
   optionally `ENABLE_TTS=true`. These are encrypted and not visible in the repo.
4. Deploy. Render builds the Docker image and serves it at
   `https://<your-service-name>.onrender.com` — a real HTTPS endpoint, so the
   browser's mic permission prompt works exactly like on localhost. Share
   that URL with teammates.

Free-tier tradeoffs worth knowing before a live demo: the service sleeps
after 15 minutes of inactivity (30-60s cold start on the next request), and
512MB RAM is tight for `faster-whisper` — the `Dockerfile` already defaults
`LOCAL_WHISPER_MODEL` to `base` for that reason; drop to `tiny` via an
environment variable on Render if it still OOMs.

## Architecture

Everything runs from **one file**, `app.py` — FastAPI backend (STT benchmark,
agent intent parsing + wallet execution, optional TTS), the mock wallet, and
the frontend (inline HTML/JS, no build step, no separate static files) all
live there so a teammate can run the whole project with one `pip install` and
one `uvicorn` command. `benchmark/run_benchmark.py` is the only other code
file, and it imports straight from `app.py` to guarantee the report reflects
the same code path as the live demo.

## Ethics

See `ETHICS.md`.
