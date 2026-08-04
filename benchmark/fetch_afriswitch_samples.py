"""One-off helper: pull real, reference-labeled code-switched clips from
Intron's own AfriSwitch dataset (huggingface.co/datasets/intronhealth/AfriSwitch)
into benchmark/samples/<lang>/, so run_benchmark.py has real ground-truth audio
to score against instead of clips you had to record yourself.

AfriSwitch covers Pidgin and Swahili but NOT Akan/Twi — you'll still need to
source or record Akan clips separately (see ETHICS.md re: consent) and drop
them into benchmark/samples/ak/.

One-time setup (not needed for the main app, so kept out of requirements.txt):
    pip install datasets soundfile
    huggingface-cli login   # or: hf auth login
Then accept AfriSwitch's access terms once, logged in as the same account, at:
    https://huggingface.co/datasets/intronhealth/AfriSwitch

Usage:
    python benchmark/fetch_afriswitch_samples.py [--n 5]
"""

import argparse
import os

from datasets import load_dataset
import soundfile as sf

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

# AfriSwitch config name -> our language code / folder.
AFRISWITCH_CONFIGS = {"pidgin": "pcm", "swahili": "sw"}


def fetch(config: str, lang_code: str, n: int):
    out_dir = os.path.join(SAMPLES_DIR, lang_code)
    os.makedirs(out_dir, exist_ok=True)

    try:
        ds = load_dataset("intronhealth/AfriSwitch", config, split="test", streaming=True)
    except Exception as exc:
        print(f"Could not load config '{config}': {exc}")
        print("Check https://huggingface.co/datasets/intronhealth/AfriSwitch for the exact config names,")
        print("and make sure you've accepted the dataset's access terms while logged in via `hf auth login`.")
        return

    count = 0
    for example in ds:
        if count >= n:
            break
        audio = example["audio"]
        transcript = example.get("transcription", "").strip()
        if not transcript:
            continue
        stem = f"afriswitch_{config}_{count:03d}"
        sf.write(os.path.join(out_dir, f"{stem}.wav"), audio["array"], audio["sampling_rate"])
        with open(os.path.join(out_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)
        count += 1

    print(f"Wrote {count} {config} clip(s) to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="clips per language")
    args = parser.parse_args()

    for config, lang_code in AFRISWITCH_CONFIGS.items():
        fetch(config, lang_code, args.n)

    print("\nNote: AfriSwitch has no Akan/Twi config — add Akan clips to benchmark/samples/ak/ yourself.")


if __name__ == "__main__":
    main()
