# Ethics, privacy & inclusion note

**Financial data.** All wallet balances, contacts, and transactions are mock
data held in server memory (`app.py`) and reset on restart or via the "Reset
Wallet" button. No real money, accounts, or third-party financial systems are
touched.

**Audio handling.** Recorded clips are sent to the backend, forwarded to the
three STT engines for transcription, written to a temp file for the duration
of that request, and deleted immediately after (`app.py`, `api_transcribe`).
Nothing is persisted or logged by default. If a team member wants to
contribute a clip to `benchmark/samples/` for the benchmark report, that is a
deliberate, separate action — never an automatic side effect of using the app
— and should only be done with the speaker's consent, using their own voice
or a clip they have rights to share.

**Consent & representation.** Akan, Pidgin, and Swahili speakers, like
speakers of most African languages, are underserved by mainstream voice AI
trained primarily on high-resource languages. This project exists to make
that gap visible (via the benchmark) and to demonstrate a concrete path to
closing it for a real customer-service use case, not to collect or profit
from anyone's voice data without their knowledge.

**Bias awareness.** The Groq Whisper and local faster-whisper baselines are
included precisely because they are *not* tuned for Akan or Pidgin (Swahili
is Whisper-supported, so it's a fairer fight there) — their expected weaker
performance on code-switched Akan/Pidgin audio is the finding, not a flaw in
the benchmark. We report all three transcripts verbatim, including failures,
rather than cherry-picking favorable runs.

**Safety.** The mock wallet enforces basic guardrails (no negative amounts, no
sends to unknown contacts, no overdrafts) so a misheard transcript can't
silently corrupt state in a way that would look like a real financial error.
