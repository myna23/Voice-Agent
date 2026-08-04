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

## How it works

1. **Pick a language** (Akan / Pidgin / Swahili) and **record** a spoken request in the browser.
2. **Transcribe** the same clip through three STT engines concurrently:
   - **Intron Sahara** (`use_language_asr_input` = `ak` / `pcm` / `sw`) — language-specialized, paid (challenge-provided key).
   - **Groq Whisper-large-v3** — free tier, global commercial-grade, not tuned for these languages.
   - **Local `faster-whisper`** — open-source, runs offline, not tuned for these languages.
3. **Act**: the Sahara transcript goes to Groq's `llama-3.3-70b-versatile`
   (free tier, function-calling) to classify intent (`check_balance` /
   `send_money` / `buy_bundle` / `file_complaint`) and extract slots. The
   backend then **executes** it against an in-memory mock wallet.
4. **Confirm**: a natural-language confirmation is shown, with an optional
   spoken confirmation via Sahara TTS (feature-flagged, falls back to
   text-only if TTS isn't ready in time).

Only the Sahara calls cost anything — STT baseline #2 and the agent LLM both
run on Groq's free tier via the OpenAI-compatible `openai` SDK pointed at
Groq's `base_url`, and `faster-whisper` runs locally.

Groq and local faster-whisper are included specifically *because* they
aren't tuned for Akan/Pidgin/Swahili — the benchmark is meant to show Sahara
winning on code-switched audio. Run `benchmark/run_benchmark.py` (below) to
generate `benchmark/report.md` from real runs, not hand-picked output.

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
pip install datasets soundfile   # one-time, only for this script
python benchmark/fetch_afriswitch_samples.py   # pulls real Pidgin/Swahili clips + reference transcripts
python benchmark/run_benchmark.py
```

`fetch_afriswitch_samples.py` pulls labeled clips from Intron's own
[AfriSwitch dataset](https://huggingface.co/datasets/intronhealth/AfriSwitch)
into `benchmark/samples/pcm/` and `benchmark/samples/sw/` — real ground truth,
not self-recorded guesses. It's gated: accept the dataset's terms once at
that link while logged in as the same account as `hf auth login`. **AfriSwitch
has no Akan/Twi config**, so Akan clips still need to be recorded/sourced
separately (with consent — see `ETHICS.md`) into `benchmark/samples/ak/`, each
`clip.wav` paired with a `clip.txt` reference transcript.

`run_benchmark.py` scores WER and CER (raw + normalized), matching
[Intron's own benchmarking methodology](https://github.com/intron-innovation/Intron-Multimodal-Benchmarking),
and writes `benchmark/report.md`.

## Deploying so teammates can test it

Render's free web service tier (512MB RAM, no credit card required).

1. Push this project to a GitHub repo (you run the `git push` yourself):
   ```bash
   git init   # if this folder isn't already a git repo
   git add .
   git commit -m "Sahara Pay Voice Agent"
   git remote add origin https://github.com/<your-username>/sahara-voice-agent.git
   git push -u origin main
   ```
2. At [render.com](https://render.com), **New → Web Service** → connect the
   repo. Render auto-detects the `Dockerfile` — no other config needed.
3. Under **Environment**, add: `INTRON_API_KEY`, `GROQ_API_KEY`, optionally `ENABLE_TTS=true`.
4. Deploy. Serves at `https://<your-service-name>.onrender.com` — share that
   URL with teammates.

Free-tier tradeoffs: sleeps after 15 minutes of inactivity (30-60s cold start
on the next request), and 512MB RAM is tight for `faster-whisper` — the
`Dockerfile` defaults `LOCAL_WHISPER_MODEL` to `base` for that reason; drop
to `tiny` via an environment variable on Render if it still OOMs.

## Architecture

Everything runs from **one file**, `app.py` — FastAPI backend, mock wallet,
and frontend (inline HTML/JS, no build step) all live there. One
`pip install` + one `uvicorn` command runs the whole project.
`benchmark/run_benchmark.py` is the only other code file, and imports
straight from `app.py` so the report reflects the same code path as the demo.

## Ethics

See `ETHICS.md`.
