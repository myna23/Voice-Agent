"""Sahara Pay Voice Agent — single-file app.

Flow: record audio -> /api/transcribe (3 STT engines, run concurrently, for
benchmarking + demo) -> /api/act (LLM intent parsing, executes a real action
against an in-memory mock wallet) -> optional /api/tts (spoken confirmation).

Everything — backend, mock wallet, and frontend — lives in this one file on
purpose, so a teammate can run the whole thing with:

    pip install -r requirements.txt
    uvicorn app:app --reload

STT engines benchmarked on the same clip:
  - Intron Sahara       — the language-specialized engine (Akan / Pidgin / Swahili)
  - Groq Whisper-large-v3 — global, free-tier commercial engine, not tuned for these languages
  - Local faster-whisper  — open-source/offline engine, not tuned for these languages

Agent brain: Groq's llama-3.3-70b-versatile (free tier), via the OpenAI-compatible
`openai` SDK pointed at Groq's base_url — so only one extra package is needed
and it's free to run.
"""

import asyncio
import json
import os
import tempfile
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from openai import AsyncOpenAI

load_dotenv()

INTRON_API_KEY = os.environ.get("INTRON_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LOCAL_WHISPER_MODEL = os.environ.get("LOCAL_WHISPER_MODEL", "small")
ENABLE_TTS = os.environ.get("ENABLE_TTS", "false").lower() == "true"
TTS_TIMEOUT_SECONDS = float(os.environ.get("TTS_TIMEOUT_SECONDS", "8"))

INTRON_BASE_URL = "https://infer.voice.intron.io"

# Sahara language code -> display name, Whisper ISO-639-1 hint (STT), and
# Sahara TTS accent (per docs.voice.intron.io/docs/tts/supported-languages-and-accents).
# Whisper's training set doesn't include Akan or Pidgin as distinct languages,
# so we don't pass a language hint for those — Groq/local Whisper will guess,
# which is expected to perform worse than Sahara. That contrast is the point
# of the benchmark. Swahili IS a Whisper-supported language, so we hint it.
# Sahara TTS has no Akan voice at all (confirmed against their docs), so
# tts_accent is None there — /api/tts skips the call rather than guessing.
LANGUAGES = {
    "ak": {"name": "Akan", "whisper_hint": None, "tts_accent": None},
    "pcm": {"name": "Pidgin", "whisper_hint": None, "tts_accent": "pidgin"},
    "sw": {"name": "Swahili", "whisper_hint": "sw", "tts_accent": "swahili"},
}

app = FastAPI(title="Sahara Pay Voice Agent")
groq_client = (
    AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
)

_local_whisper_model = None  # lazy-loaded on first use, shared across requests


def _get_local_whisper():
    global _local_whisper_model
    if _local_whisper_model is None:
        from faster_whisper import WhisperModel

        _local_whisper_model = WhisperModel(LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8")
    return _local_whisper_model


# ==========================================================================
# Mock wallet — stands in for a real telco/fintech backend so the agent has
# something concrete to act on. Swap for real API calls to go to production.
# ==========================================================================

STARTING_BALANCE = 250.00

BUNDLE_PRICES = {
    "1gb data": 5.00,
    "5gb data": 20.00,
    "10 cedis airtime": 10.00,
    "20 cedis airtime": 20.00,
}

_wallet = {
    "balance": STARTING_BALANCE,
    "contacts": {"ama": 0.0, "kofi": 0.0, "chidi": 0.0, "amina": 0.0},
    "bundles": [],
    "complaints": [],
}


def wallet_reset():
    _wallet["balance"] = STARTING_BALANCE
    for name in _wallet["contacts"]:
        _wallet["contacts"][name] = 0.0
    _wallet["bundles"].clear()
    _wallet["complaints"].clear()
    return wallet_snapshot()


def wallet_snapshot():
    return {
        "balance": round(_wallet["balance"], 2),
        "contacts": dict(_wallet["contacts"]),
        "bundles": list(_wallet["bundles"]),
        "complaints": list(_wallet["complaints"]),
    }


def check_balance():
    return {"ok": True, "result": {"balance": round(_wallet["balance"], 2)},
            "message": f"Your balance is {_wallet['balance']:.2f} cedis."}


def top_up(amount: float):
    if amount is None or amount <= 0:
        return {"ok": False, "result": None, "message": "I need a valid top-up amount greater than zero."}
    _wallet["balance"] += amount
    return {"ok": True, "result": {"balance": round(_wallet["balance"], 2), "amount": amount},
            "message": f"Added {amount:.2f} cedis. New balance: {_wallet['balance']:.2f} cedis."}


def send_money(to: str, amount: float):
    to_key = (to or "").strip().lower()
    if to_key not in _wallet["contacts"]:
        return {"ok": False, "result": None,
                "message": f"I don't recognize '{to}' as a contact. Known contacts: "
                + ", ".join(n.title() for n in _wallet["contacts"])}
    if amount is None or amount <= 0:
        return {"ok": False, "result": None, "message": "I need a valid amount greater than zero."}
    if amount > _wallet["balance"]:
        return {"ok": False, "result": None,
                "message": f"Insufficient balance. You have {_wallet['balance']:.2f} cedis, "
                f"tried to send {amount:.2f}."}
    _wallet["balance"] -= amount
    _wallet["contacts"][to_key] += amount
    return {"ok": True, "result": {"balance": round(_wallet["balance"], 2), "sent_to": to_key, "amount": amount},
            "message": f"Sent {amount:.2f} cedis to {to_key.title()}. New balance: {_wallet['balance']:.2f} cedis."}


def buy_bundle(product: str, amount: float | None = None):
    product_key = (product or "").strip().lower()
    price = BUNDLE_PRICES.get(product_key)
    if price is None and amount:
        price = amount
    if price is None:
        return {"ok": False, "result": None,
                "message": f"I don't recognize '{product}'. Available: " + ", ".join(BUNDLE_PRICES)}
    if price > _wallet["balance"]:
        return {"ok": False, "result": None,
                "message": f"Insufficient balance. You have {_wallet['balance']:.2f} cedis, "
                f"'{product_key}' costs {price:.2f}."}
    _wallet["balance"] -= price
    _wallet["bundles"].append(product_key)
    return {"ok": True, "result": {"balance": round(_wallet["balance"], 2), "bundle": product_key, "price": price},
            "message": f"Purchased {product_key} for {price:.2f} cedis. New balance: {_wallet['balance']:.2f} cedis."}


def file_complaint(text: str):
    text = (text or "").strip()
    if not text:
        return {"ok": False, "result": None, "message": "I didn't catch what the complaint was about."}
    _wallet["complaints"].append(text)
    ticket_id = f"CX-{len(_wallet['complaints']):04d}"
    return {"ok": True, "result": {"ticket_id": ticket_id, "text": text},
            "message": f"Logged your complaint as ticket {ticket_id}. Our team will follow up."}


# ==========================================================================
# STT engines — each returns {"engine", "label", "ok", "transcript", "latency_ms", "error"}
# ==========================================================================

def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _engine_error(engine: str, label: str, error: str, start: float) -> dict:
    return {"engine": engine, "label": label, "ok": False, "transcript": "", "latency_ms": _ms(start), "error": error}


async def transcribe_sahara(audio_path: str, filename: str, lang: str) -> dict:
    label = f"Intron Sahara ({LANGUAGES[lang]['name']}-specialized)"
    start = time.monotonic()
    if not INTRON_API_KEY:
        return _engine_error("sahara", label, "INTRON_API_KEY not set", start)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(
                    f"{INTRON_BASE_URL}/file/v1/upload/sync",
                    headers={"Authorization": f"Bearer {INTRON_API_KEY}"},
                    data={"audio_file_name": filename, "use_language_asr_input": lang},
                    files={"audio_file_blob": (filename, f)},
                )
        resp.raise_for_status()
        transcript = resp.json().get("data", {}).get("audio_transcript", "")
        return {"engine": "sahara", "label": label, "ok": True, "transcript": transcript,
                "latency_ms": _ms(start), "error": None}
    except Exception as exc:  # noqa: BLE001 — surface any engine failure to the UI, don't crash the request
        return _engine_error("sahara", label, str(exc), start)


async def transcribe_groq(audio_path: str, lang: str) -> dict:
    label = "Groq Whisper-large-v3 (free, global, not tuned for this language)"
    start = time.monotonic()
    if not groq_client:
        return _engine_error("groq_whisper", label, "GROQ_API_KEY not set", start)
    try:
        kwargs = {"model": "whisper-large-v3", "file": open(audio_path, "rb")}
        hint = LANGUAGES[lang]["whisper_hint"]
        if hint:
            kwargs["language"] = hint
        resp = await groq_client.audio.transcriptions.create(**kwargs)
        return {"engine": "groq_whisper", "label": label, "ok": True, "transcript": resp.text,
                "latency_ms": _ms(start), "error": None}
    except Exception as exc:  # noqa: BLE001
        return _engine_error("groq_whisper", label, str(exc), start)


async def transcribe_local_whisper(audio_path: str, lang: str) -> dict:
    label = f"Local faster-whisper ({LOCAL_WHISPER_MODEL}, open-source, offline, not tuned for this language)"
    start = time.monotonic()
    hint = LANGUAGES[lang]["whisper_hint"]  # same Whisper-family language codes as Groq; None left to auto-detect
    try:
        def _run():
            model = _get_local_whisper()
            segments, _info = model.transcribe(audio_path, language=hint)
            return " ".join(seg.text.strip() for seg in segments)

        transcript = await asyncio.to_thread(_run)
        return {"engine": "local_whisper", "label": label, "ok": True, "transcript": transcript,
                "latency_ms": _ms(start), "error": None}
    except Exception as exc:  # noqa: BLE001
        return _engine_error("local_whisper", label, str(exc), start)


MIN_AUDIO_BYTES = 3000  # a valid webm/opus clip of even ~1s is well above this


@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile, lang: str = Form("ak")):
    if lang not in LANGUAGES:
        lang = "ak"
    filename = audio.filename or "clip.wav"
    suffix = os.path.splitext(filename)[1] or ".wav"
    content = await audio.read()

    if len(content) < MIN_AUDIO_BYTES:
        error = {
            "ok": False, "transcript": "", "latency_ms": 0,
            "error": f"Recording too short/empty ({len(content)} bytes) — record for at least 1-2 seconds and check mic permissions.",
        }
        return JSONResponse({
            "engines": [
                {"engine": "sahara", "label": "Intron Sahara", **error},
                {"engine": "groq_whisper", "label": "Groq Whisper-large-v3", **error},
                {"engine": "local_whisper", "label": "Local faster-whisper", **error},
            ],
            "primary_transcript": "",
            "lang": lang,
        })

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = await asyncio.gather(
            transcribe_sahara(tmp_path, filename, lang),
            transcribe_groq(tmp_path, lang),
            transcribe_local_whisper(tmp_path, lang),
        )
    finally:
        os.unlink(tmp_path)

    primary = next((r["transcript"] for r in results if r["engine"] == "sahara" and r["ok"]), None)
    if not primary:
        primary = next((r["transcript"] for r in results if r["ok"]), "")

    return JSONResponse({"engines": results, "primary_transcript": primary, "lang": lang})


# ==========================================================================
# Agent — parses intent from a (possibly code-switched) transcript and
# actually executes it against the mock wallet.
# ==========================================================================

INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_wallet_request",
        "description": "Classify a mobile-money/telco customer request and extract its parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["check_balance", "send_money", "buy_bundle", "top_up", "file_complaint", "unknown"],
                },
                "to": {"type": ["string", "null"], "description": "Recipient name for send_money"},
                "amount": {"type": ["number", "null"], "description": "Amount in cedis, for send_money, buy_bundle, or top_up"},
                "product": {
                    "type": ["string", "null"],
                    "description": "For buy_bundle, must be exactly one of the catalog names given in the system prompt "
                    "(e.g. a caller saying '1 gigabyte of data' or 'one gig' should map to '1gb data').",
                },
                "complaint_text": {"type": ["string", "null"], "description": "Summary of the issue for file_complaint"},
            },
            "required": ["intent", "to", "amount", "product", "complaint_text"],
        },
    },
}

SYSTEM_PROMPT = (
    "You classify customer-care voice requests for a mobile money/telco service. Callers often "
    "code-switch between English and Akan, Nigerian/Ghanaian Pidgin, or Swahili in the same sentence, "
    "and amounts/products are frequently said in English even in an otherwise local-language sentence. "
    "'top_up' means the caller wants to ADD money to their own balance (e.g. 'reload my wallet with 50 cedis', "
    "'top up my account', 'I want to add 20 cedis') — distinct from send_money, which sends money to someone else. "
    "For buy_bundle requests, the 'product' field must exactly match one of these catalog names, "
    "however the caller phrased it (e.g. '1 gigabyte of data', 'one gig', 'a gig of data' all mean '1gb data'): "
    + ", ".join(f"'{p}'" for p in BUNDLE_PRICES)
    + ". If a spoken request doesn't clearly match any catalog item, leave product null rather than guessing. "
    "Always call the classify_wallet_request function with your best interpretation, even if the "
    "transcript is noisy or partially mis-transcribed."
)


@app.post("/api/act")
async def api_act(payload: dict):
    transcript = (payload or {}).get("transcript", "").strip()
    if not transcript:
        return JSONResponse({"ok": False, "message": "No transcript provided."}, status_code=400)
    if not groq_client:
        return JSONResponse({"ok": False, "message": "GROQ_API_KEY not set."}, status_code=500)

    completion = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        tools=[INTENT_TOOL],
        tool_choice={"type": "function", "function": {"name": "classify_wallet_request"}},
    )
    call = completion.choices[0].message.tool_calls[0]
    parsed = json.loads(call.function.arguments)
    intent = parsed.get("intent")

    if intent == "check_balance":
        outcome = check_balance()
    elif intent == "send_money":
        outcome = send_money(parsed.get("to"), parsed.get("amount"))
    elif intent == "buy_bundle":
        outcome = buy_bundle(parsed.get("product"), parsed.get("amount"))
    elif intent == "top_up":
        outcome = top_up(parsed.get("amount"))
    elif intent == "file_complaint":
        outcome = file_complaint(parsed.get("complaint_text"))
    else:
        outcome = {"ok": False, "result": None, "message": "Sorry, I couldn't understand that request."}

    return JSONResponse({"transcript": transcript, "parsed": parsed, **outcome, "wallet": wallet_snapshot()})


@app.post("/api/reset")
async def api_reset():
    return JSONResponse(wallet_reset())


# ==========================================================================
# Optional TTS confirmation — never blocks the core flow.
# ==========================================================================

@app.post("/api/tts")
async def api_tts(payload: dict):
    if not ENABLE_TTS:
        return JSONResponse({"ok": False, "reason": "tts_disabled"})
    text = (payload or {}).get("text", "").strip()
    lang = (payload or {}).get("lang", "ak")
    if lang not in LANGUAGES:
        lang = "ak"
    if not text or not INTRON_API_KEY:
        return JSONResponse({"ok": False, "reason": "missing_text_or_key"})

    tts_accent = LANGUAGES[lang]["tts_accent"]
    if not tts_accent:
        # Sahara has no TTS voice for this language (e.g. Akan) — skip the call
        # rather than sending a guessed accent value that's guaranteed to fail.
        return JSONResponse({"ok": False, "reason": "tts_unsupported_for_language"})

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            enqueue = await client.post(
                f"{INTRON_BASE_URL}/tts/v1/enqueue",
                headers={"Authorization": f"Bearer {INTRON_API_KEY}", "Content-Type": "application/json"},
                json={"text": text, "voice_language": lang, "voice_accent": tts_accent,
                      "voice_gender": "female"},
            )
            enqueue.raise_for_status()
            text_id = enqueue.json()["data"]["text_id"]

            deadline = time.monotonic() + TTS_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                status_resp = await client.get(
                    f"{INTRON_BASE_URL}/tts/v1/status/{text_id}",
                    headers={"Authorization": f"Bearer {INTRON_API_KEY}"},
                )
                status_resp.raise_for_status()
                body = status_resp.json()
                status = body.get("data", {}).get("status") or body.get("status")
                if status == "TTS_TEXT_AUDIO_GENERATED":
                    audio_url = (body.get("data") or {}).get("audio_url") or (body.get("data") or {}).get("url")
                    if audio_url:
                        return JSONResponse({"ok": True, "audio_url": audio_url})
                    return JSONResponse({"ok": False, "reason": "generated_but_no_url"})
                if status == "TTS_TEXT_AUDIO_PROCESSING_FAILED":
                    return JSONResponse({"ok": False, "reason": "tts_processing_failed"})
                await asyncio.sleep(1)
            return JSONResponse({"ok": False, "reason": "timeout"})
    except Exception as exc:  # noqa: BLE001 — TTS is a bonus; never raise past this point
        return JSONResponse({"ok": False, "reason": str(exc)})


# ==========================================================================
# Frontend — single inline page, no build step, no separate static files.
# ==========================================================================

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Sahara Pay Voice Agent</title>
<style>
  :root { color-scheme: light dark; --bg:#0b0f14; --panel:#131a22; --border:#253140; --text:#e6edf3;
    --muted:#8b98a5; --accent:#5b8cff; --ok:#3ecf8e; --err:#ff6b6b; }
  * { box-sizing: border-box; }
  body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  .wrap { max-width:880px; margin:0 auto; }
  h1 { font-size:1.4rem; margin:0 0 0.25rem; }
  .sub { color:var(--muted); margin:0 0 1.75rem; font-size:0.92rem; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:1.25rem; margin-bottom:1rem; }
  .record-row { display:flex; align-items:center; gap:1rem; flex-wrap:wrap; }
  select { font:inherit; background:#0e141b; color:var(--text); border:1px solid var(--border); border-radius:8px; padding:0.55rem 0.75rem; }
  button { font:inherit; border:none; border-radius:8px; padding:0.65rem 1.1rem; cursor:pointer; font-weight:600; }
  #recordBtn { background:var(--accent); color:white; }
  #recordBtn.recording { background:var(--err); }
  #resetBtn { background:transparent; color:var(--muted); border:1px solid var(--border); }
  #status { color:var(--muted); font-size:0.9rem; }
  .engines { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:0.75rem; }
  .engine { background:#0e141b; border:1px solid var(--border); border-radius:10px; padding:0.85rem; font-size:0.88rem; }
  .engine h3 { margin:0 0 0.4rem; font-size:0.82rem; color:var(--muted); font-weight:600; }
  .engine .transcript { color:var(--text); min-height:2.4em; white-space:pre-wrap; }
  .engine .meta { margin-top:0.5rem; font-size:0.76rem; color:var(--muted); }
  .engine.err .transcript { color:var(--err); font-size:0.8rem; }
  .badge { display:inline-block; padding:0.1rem 0.5rem; border-radius:999px; font-size:0.72rem; font-weight:700; }
  .badge.ok { background:rgba(62,207,142,0.15); color:var(--ok); }
  .badge.err { background:rgba(255,107,107,0.15); color:var(--err); }
  .action-result { font-size:0.95rem; line-height:1.5; }
  .action-result.ok { color:var(--ok); }
  .action-result.err { color:var(--err); }
  pre.wallet { background:#0e141b; border:1px solid var(--border); border-radius:8px; padding:0.75rem; font-size:0.8rem; overflow-x:auto; }
  .hint { color:var(--muted); font-size:0.82rem; }
  audio { width:100%; margin-top:0.5rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Sahara Pay Voice Agent</h1>
  <p class="sub">Pick the local language you'll code-switch with English, then speak a request — e.g. "check my balance", "send 20 cedis to Ama", "buy me 1GB data", "top up my wallet with 50 cedis". Every clip is benchmarked across three speech models before the agent acts.</p>

  <div class="card">
    <div class="record-row">
      <select id="langSelect">
        <option value="ak">Akan</option>
        <option value="pcm">Pidgin</option>
        <option value="sw">Swahili</option>
      </select>
      <button id="recordBtn">Start Recording</button>
      <button id="resetBtn" title="Wipes the wallet back to the starting demo balance — for testing between takes, not something a real user would do">Reset Wallet (demo only)</button>
      <span id="status">Idle</span>
    </div>
  </div>

  <div class="card" id="enginesCard" style="display:none;">
    <h2 style="margin-top:0;font-size:1rem;">Speech-to-text benchmark</h2>
    <div class="engines" id="engines"></div>
  </div>

  <div class="card" id="actionCard" style="display:none;">
    <h2 style="margin-top:0;font-size:1rem;">Agent action</h2>
    <div class="action-result" id="actionResult"></div>
    <audio id="ttsAudio" controls style="display:none;"></audio>
  </div>

  <div class="card">
    <h2 style="margin-top:0;font-size:1rem;">Wallet state</h2>
    <pre class="wallet" id="walletState">Not loaded yet.</pre>
    <p class="hint">Mock data only — no real money moves. See ETHICS.md.</p>
  </div>
</div>

<script>
const recordBtn = document.getElementById('recordBtn');
const resetBtn = document.getElementById('resetBtn');
const langSelect = document.getElementById('langSelect');
const statusEl = document.getElementById('status');
const enginesCard = document.getElementById('enginesCard');
const enginesEl = document.getElementById('engines');
const actionCard = document.getElementById('actionCard');
const actionResultEl = document.getElementById('actionResult');
const ttsAudio = document.getElementById('ttsAudio');
const walletStateEl = document.getElementById('walletState');

let mediaRecorder, chunks = [], recording = false;

function setStatus(text) { statusEl.textContent = text; }

recordBtn.addEventListener('click', async () => {
  if (!recording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        handleClip(blob);
      };
      mediaRecorder.start();
      recording = true;
      recordBtn.textContent = 'Stop Recording';
      recordBtn.classList.add('recording');
      setStatus('Recording…');
    } catch (err) {
      setStatus('Mic access denied: ' + err.message);
    }
  } else {
    mediaRecorder.stop();
    recording = false;
    recordBtn.textContent = 'Start Recording';
    recordBtn.classList.remove('recording');
    setStatus('Processing…');
  }
});

resetBtn.addEventListener('click', async () => {
  const resp = await fetch('/api/reset', { method: 'POST' });
  walletStateEl.textContent = JSON.stringify(await resp.json(), null, 2);
  actionCard.style.display = 'none';
  enginesCard.style.display = 'none';
  setStatus('Wallet reset.');
});

async function handleClip(blob) {
  enginesCard.style.display = 'block';
  actionCard.style.display = 'none';
  enginesEl.innerHTML = '<p class="hint">Transcribing with Sahara, Groq Whisper, and local Whisper…</p>';

  const form = new FormData();
  form.append('audio', blob, 'clip.webm');
  form.append('lang', langSelect.value);

  let transcribeData;
  try {
    const resp = await fetch('/api/transcribe', { method: 'POST', body: form });
    transcribeData = await resp.json();
  } catch (err) {
    setStatus('Transcription request failed: ' + err.message);
    return;
  }

  renderEngines(transcribeData.engines);

  const primary = transcribeData.primary_transcript;
  if (!primary) {
    setStatus('No engine produced a transcript — try again.');
    return;
  }

  setStatus('Interpreting request…');
  const actResp = await fetch('/api/act', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript: primary }),
  });
  const actData = await actResp.json();
  renderAction(actData);
  setStatus('Done.');
}

function renderEngines(engines) {
  enginesEl.innerHTML = '';
  for (const e of engines) {
    const div = document.createElement('div');
    div.className = 'engine' + (e.ok ? '' : ' err');
    div.innerHTML = `
      <h3>${e.label}</h3>
      <div class="transcript">${e.ok ? escapeHtml(e.transcript || '(empty)') : escapeHtml(e.error)}</div>
      <div class="meta">
        <span class="badge ${e.ok ? 'ok' : 'err'}">${e.ok ? 'OK' : 'ERROR'}</span>
        ${e.latency_ms}ms
      </div>`;
    enginesEl.appendChild(div);
  }
}

function renderAction(data) {
  actionCard.style.display = 'block';
  actionResultEl.className = 'action-result ' + (data.ok ? 'ok' : 'err');
  actionResultEl.textContent = data.message || '(no response)';
  if (data.wallet) {
    walletStateEl.textContent = JSON.stringify(data.wallet, null, 2);
  }
  maybePlayTts(data.message);
}

async function maybePlayTts(text) {
  if (!text) return;
  try {
    const resp = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, lang: langSelect.value }),
    });
    const data = await resp.json();
    if (data.ok && data.audio_url) {
      ttsAudio.src = data.audio_url;
      ttsAudio.style.display = 'block';
      ttsAudio.play().catch(() => {});
    } else {
      ttsAudio.style.display = 'none';
    }
  } catch {
    ttsAudio.style.display = 'none';
  }
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

fetch('/api/reset', { method: 'POST' })
  .then((r) => r.json())
  .then((data) => { walletStateEl.textContent = JSON.stringify(data, null, 2); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML
