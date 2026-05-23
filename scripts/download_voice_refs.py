"""Tải mẫu giọng tiếng Việt đa dạng từ VIVOS dataset → `data/refs/`.

Mỗi reference: 1 file .wav (16kHz mono, 4-8s) + 1 file .txt transcript verbatim.

VIVOS (AILAB-VNUHCM/vivos) là dataset speech VN học thuật, ~46 speakers, mỗi
speaker đọc nhiều câu rõ ràng → lý tưởng làm reference cho voice cloning.

Cách chạy:
    python scripts/download_voice_refs.py --n-speakers 12 --min-dur 4 --max-dur 8
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
REFS_DIR = BASE_DIR / "data" / "refs"
CACHE_DIR = BASE_DIR / "data" / "hf_cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="AILAB-VNUHCM/vivos", help="HF dataset id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-speakers", type=int, default=12, help="số speaker khác nhau")
    ap.add_argument("--per-speaker", type=int, default=1, help="số clip mỗi speaker")
    ap.add_argument("--min-dur", type=float, default=4.0)
    ap.add_argument("--max-dur", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    from datasets import load_dataset

    print(f"[load] {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split, cache_dir=str(CACHE_DIR))
    print(f"[load] {len(ds)} samples")

    cols = ds.column_names
    print(f"[cols] {cols}")
    audio_col = "audio"
    text_col = "sentence" if "sentence" in cols else ("text" if "text" in cols else cols[1])
    speaker_col = None
    for cand in ("speaker_id", "speaker", "client_id"):
        if cand in cols:
            speaker_col = cand
            break
    print(f"[cols] text={text_col} speaker={speaker_col}")

    # Group candidates by speaker after filtering by duration.
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for idx in tqdm(range(len(ds)), desc="scan"):
        info = ds[idx][audio_col]
        try:
            dur = len(info["array"]) / float(info["sampling_rate"])
        except Exception:
            continue
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        text = (ds[idx].get(text_col) or "").strip()
        if not text or len(text.split()) < 5:
            continue
        spk = ds[idx].get(speaker_col, f"row{idx}") if speaker_col else f"row{idx}"
        by_speaker[spk].append(idx)

    speakers = list(by_speaker.keys())
    random.shuffle(speakers)
    speakers = speakers[: args.n_speakers]
    print(f"[pick] {len(speakers)} speakers (out of {len(by_speaker)})")

    REFS_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for spk in speakers:
        rows = by_speaker[spk][: args.per_speaker]
        for j, row in enumerate(rows):
            sample = ds[row]
            audio = sample[audio_col]
            text = sample[text_col].strip()
            name = f"{spk}_{j}".replace("/", "_").replace(" ", "_")
            wav_path = REFS_DIR / f"{name}.wav"
            txt_path = REFS_DIR / f"{name}.txt"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"], subtype="PCM_16")
            txt_path.write_text(text, encoding="utf-8")
            index.append({
                "id": name,
                "wav": str(wav_path),
                "txt": str(txt_path),
                "text": text,
                "speaker": spk,
                "sample_rate": audio["sampling_rate"],
                "duration": round(len(audio["array"]) / float(audio["sampling_rate"]), 3),
            })

    (REFS_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {len(index)} refs -> {REFS_DIR}")


if __name__ == "__main__":
    main()
