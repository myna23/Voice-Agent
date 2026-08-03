"""Batch benchmark for the challenge's required Benchmark Report deliverable.

Usage:
    python benchmark/run_benchmark.py

Organize code-switched clips by language under benchmark/samples/<lang>/, where
<lang> is one of: ak (Akan), pcm (Pidgin), sw (Swahili) — e.g.
benchmark/samples/ak/greeting.wav. Clips dropped directly in benchmark/samples/
(no language subfolder) are assumed to be Akan.

Optionally add a matching <clip_name>.txt next to a clip with its reference
transcript to get a rough word-overlap accuracy score; without it, only
latency and raw transcripts are reported.

Writes benchmark/report.md. Reuses the exact same engine functions as the live
app (app.py) so the report reflects real, reproducible runs — not hand-picked
demo outputs.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import app as sahara_app  # noqa: E402

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "report.md")
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}


def word_overlap_score(hyp: str, ref: str) -> float:
    hyp_words = hyp.lower().split()
    ref_words = ref.lower().split()
    if not ref_words:
        return float("nan")
    ref_set = list(ref_words)
    matches = 0
    for w in hyp_words:
        if w in ref_set:
            ref_set.remove(w)
            matches += 1
    return matches / len(ref_words)


def find_clips():
    """Yield (path, lang) for every audio clip under SAMPLES_DIR."""
    if not os.path.isdir(SAMPLES_DIR):
        return
    for entry in sorted(os.listdir(SAMPLES_DIR)):
        full = os.path.join(SAMPLES_DIR, entry)
        if os.path.isdir(full):
            lang = entry if entry in sahara_app.LANGUAGES else "ak"
            for f in sorted(os.listdir(full)):
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                    yield os.path.join(full, f), lang
        elif os.path.splitext(entry)[1].lower() in AUDIO_EXTS:
            yield full, "ak"


async def run_one(path: str, lang: str) -> dict:
    filename = os.path.basename(path)
    results = await asyncio.gather(
        sahara_app.transcribe_sahara(path, filename, lang),
        sahara_app.transcribe_groq(path, lang),
        sahara_app.transcribe_local_whisper(path),
    )
    ref_path = os.path.splitext(path)[0] + ".txt"
    reference = None
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            reference = f.read().strip()
    return {"file": filename, "lang": lang, "reference": reference, "results": results}


async def main():
    clips = list(find_clips())
    if not clips:
        print(f"No audio clips found under {SAMPLES_DIR}. Add files under samples/ak/, samples/pcm/, or samples/sw/.")
        return

    print(f"Benchmarking {len(clips)} clip(s) across 3 engines…")
    all_runs = []
    for clip, lang in clips:
        print(f"  [{lang}] {os.path.basename(clip)}…")
        all_runs.append(await run_one(clip, lang))

    write_report(all_runs)
    print(f"Report written to {REPORT_PATH}")


def write_report(all_runs: list[dict]):
    lines = [
        "# Sahara Pay Voice Agent — Code-Switched STT Benchmark Report",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · {len(all_runs)} clip(s) · "
        "Sahara vs Groq Whisper-large-v3 vs local faster-whisper",
        "",
    ]

    engine_totals = {}
    for run in all_runs:
        lines.append(f"## {run['file']} ({sahara_app.LANGUAGES[run['lang']]['name']})")
        if run["reference"]:
            lines.append(f"**Reference:** {run['reference']}")
        lines.append("")
        lines.append("| Engine | OK | Latency (ms) | Transcript | Word overlap |")
        lines.append("|---|---|---|---|---|")
        for r in run["results"]:
            engine_totals.setdefault(r["engine"], {"latency": [], "n": 0, "errors": 0})
            engine_totals[r["engine"]]["n"] += 1
            if r["ok"]:
                engine_totals[r["engine"]]["latency"].append(r["latency_ms"])
            else:
                engine_totals[r["engine"]]["errors"] += 1
            score = ""
            if run["reference"] and r["ok"]:
                score = f"{word_overlap_score(r['transcript'], run['reference']):.0%}"
            transcript = r["transcript"] if r["ok"] else f"ERROR: {r['error']}"
            lines.append(f"| {r['label']} | {'yes' if r['ok'] else 'no'} | {r['latency_ms']} | {transcript} | {score} |")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Engine | Runs | Errors | Avg latency (ms) |")
    lines.append("|---|---|---|---|")
    for engine, stats in engine_totals.items():
        avg_latency = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else float("nan")
        lines.append(f"| {engine} | {stats['n']} | {stats['errors']} | {avg_latency:.0f} |")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
