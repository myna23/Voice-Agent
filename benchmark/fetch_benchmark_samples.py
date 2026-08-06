"""One-off helper: pull real, reference-labeled clips from open datasets into
benchmark/samples/<lang>/, so run_benchmark.py has real ground-truth audio to
score against instead of clips you had to record yourself.

Two sources:
  - intronhealth/AfriSwitch (Pidgin, Swahili) — genuinely code-switched
    English + local-language audio, Intron's own dataset.
  - ghanaopendata/twi-speech-text-multispeaker-16k (Akan) — real Twi audio
    with transcripts, but MONOLINGUAL (no English code-switching). AfriSwitch
    has no Akan/Twi config at all, so this is the best open ground truth
    available for Akan; it fills the "do we have real Akan audio" gap but
    does NOT demonstrate code-switching the way the Pidgin/Swahili clips do.
    For a true code-switched Akan sample, record one yourself (with consent —
    see ETHICS.md) and drop it into benchmark/samples/ak/ alongside these.

One-time setup (not needed for the main app, so kept out of requirements.txt):
    pip install datasets soundfile
    huggingface-cli login   # or: hf auth login
AfriSwitch is gated — accept its access terms once, logged in as the same
account, at: https://huggingface.co/datasets/intronhealth/AfriSwitch
(ghanaopendata/twi-speech-text-multispeaker-16k is fully open, no login needed.)

Usage:
    python benchmark/fetch_benchmark_samples.py [--n 5]
"""

import argparse
import io
import os

import soundfile as sf
from datasets import Audio, load_dataset

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

# AfriSwitch config name -> our language code / folder.
AFRISWITCH_CONFIGS = {"pidgin": "pcm", "swahili": "sw"}


def _write_clip(out_dir: str, stem: str, audio_field: dict, transcript: str):
    """audio_field is the raw {'bytes': ..., 'path': ...} dict from an
    undecoded HF Audio column — decoded here with soundfile directly, so this
    script doesn't need torchcodec/torch (which caused an arm64/x86_64 wheel
    mismatch in this venv when installed)."""
    data, samplerate = sf.read(io.BytesIO(audio_field["bytes"]))
    sf.write(os.path.join(out_dir, f"{stem}.wav"), data, samplerate)
    with open(os.path.join(out_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)


def fetch_afriswitch(config: str, lang_code: str, n: int):
    out_dir = os.path.join(SAMPLES_DIR, lang_code)
    os.makedirs(out_dir, exist_ok=True)

    try:
        ds = load_dataset("intronhealth/AfriSwitch", config, split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"Could not load AfriSwitch config '{config}': {exc}")
        print("Check https://huggingface.co/datasets/intronhealth/AfriSwitch for exact config names,")
        print("and make sure you've accepted the dataset's access terms while logged in via `hf auth login`.")
        return

    count = 0
    for example in ds:
        if count >= n:
            break
        transcript = (example.get("transcription") or "").strip()
        if not transcript:
            continue
        _write_clip(out_dir, f"afriswitch_{config}_{count:03d}", example["audio"], transcript)
        count += 1

    print(f"Wrote {count} {config} clip(s) to {out_dir}")


def fetch_twi(n: int):
    out_dir = os.path.join(SAMPLES_DIR, "ak")
    os.makedirs(out_dir, exist_ok=True)

    try:
        ds = load_dataset("ghanaopendata/twi-speech-text-multispeaker-16k", split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"Could not load the Twi dataset: {exc}")
        return

    count = 0
    for example in ds:
        if count >= n:
            break
        transcript = (example.get("text") or "").strip()
        if not transcript:
            continue
        _write_clip(out_dir, f"twi_monolingual_{count:03d}", example["audio"], transcript)
        count += 1

    print(f"Wrote {count} Akan (monolingual Twi, not code-switched) clip(s) to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="clips per language")
    args = parser.parse_args()

    for config, lang_code in AFRISWITCH_CONFIGS.items():
        fetch_afriswitch(config, lang_code, args.n)
    fetch_twi(args.n)

    print(
        "\nNote: the Akan clips are monolingual Twi (no English code-switching) — "
        "the closest open ground truth available, since AfriSwitch has no Akan config. "
        "For a genuinely code-switched Akan sample, record one yourself (with consent) "
        "into benchmark/samples/ak/ alongside these."
    )


if __name__ == "__main__":
    main()
