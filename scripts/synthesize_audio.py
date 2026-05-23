"""Synthesize audio cho mọi câu trong generated_sentences.jsonl bằng Kokoro TTS.

Audio đầu ra:
    data/audio/synth/<id>.wav

Manifest (tương thích HF datasets):
    data/manifests/synth_manifest.jsonl
    {"audio": "...wav", "text": "...", "duration": <s>, "source": "vn_augmented",
     "vietnamese_terms": [...], "category": "...", "style": "...", "context": "..."}

Cách dùng:
    python scripts/synthesize_audio.py
    python scripts/synthesize_audio.py --voices af_heart,af_bella,am_adam --limit 100
    python scripts/synthesize_audio.py --lang a --speed 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from kokoro import KPipeline
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "generated_sentences.jsonl"
DEFAULT_OUT_DIR = DATA_DIR / "audio" / "synth"
DEFAULT_MANIFEST = DATA_DIR / "manifests" / "synth_manifest.jsonl"

SAMPLE_RATE = 24000


def load_existing_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    ids = set()
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(Path(json.loads(line)["audio"]).stem)
            except Exception:
                pass
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--voices", default="af_heart,af_bella,am_adam,bf_emma,bm_george",
                    help="comma-separated Kokoro voices to cycle through")
    ap.add_argument("--lang", default="a", help="Kokoro lang_code: a=American, b=British")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = không giới hạn")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    print(f"[config] lang={args.lang} voices={voices} speed={args.speed}")

    pipeline = KPipeline(lang_code=args.lang)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_existing_ids(args.manifest)
    print(f"[resume] {len(done_ids)} samples already done, skipping")

    items = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["id"] in done_ids:
                continue
            items.append(r)
            if args.limit and len(items) >= args.limit:
                break
    print(f"[plan] {len(items)} sentences to synthesize")

    out_manifest = args.manifest.open("a", encoding="utf-8")
    n_ok = n_fail = 0

    pbar = tqdm(items, unit="sample")
    for i, r in enumerate(pbar):
        sample_id = r["id"]
        out_wav = args.out_dir / f"{sample_id}.wav"
        voice = voices[i % len(voices)]

        try:
            audio_chunks = []
            for _, _, audio in pipeline(r["sentence"], voice=voice, speed=args.speed):
                audio_chunks.append(audio)

            if not audio_chunks:
                raise ValueError("no audio generated")

            import numpy as np
            audio_data = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            sf.write(str(out_wav), audio_data, SAMPLE_RATE)
            duration = round(len(audio_data) / SAMPLE_RATE, 3)
        except Exception as e:
            pbar.write(f"[{sample_id}] ERROR: {e}")
            n_fail += 1
            continue

        rec = {
            "audio": str(out_wav.resolve()),
            "text": r["sentence"],
            "duration": duration,
            "source": "vn_augmented",
            "vietnamese_terms": r.get("vietnamese_terms", []),
            "category": r.get("category"),
            "style": r.get("seed", {}).get("style"),
            "context": r.get("seed", {}).get("context"),
            "ref_voice": voice,
        }
        out_manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_manifest.flush()
        n_ok += 1
        pbar.set_postfix(ok=n_ok, fail=n_fail)

    out_manifest.close()
    print(f"[done] ok={n_ok} fail={n_fail} -> {args.manifest}")


if __name__ == "__main__":
    main()
