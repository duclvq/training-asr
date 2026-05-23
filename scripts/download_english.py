"""Tải dataset ASR tiếng Anh và build manifest tương thích.

Mặc định: LibriSpeech `clean.dev` (~340MB, ~5h) — đủ cho dev/smoke test.
Có thể đổi sang `clean.100` (~100h), `clean.360`, `other.500` tuỳ disk space.

Cũng support Common Voice English subset (cần HF token cho CV).

Output:
    data/manifests/english_manifest.jsonl

Mỗi record:
    {"audio": "/abs/path/to.wav", "text": "...", "duration": <s>, "source": "english"}

Audio gốc có thể là flac → ta lưu đường dẫn gốc, finetune script sẽ decode bằng torchaudio/librosa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = BASE_DIR / "data" / "manifests" / "english_manifest.jsonl"
DEFAULT_CACHE = BASE_DIR / "data" / "hf_cache"


def dump_librispeech(split: str, manifest_path: Path, cache_dir: Path, max_items: int) -> None:
    from datasets import load_dataset

    print(f"[librispeech] loading split={split} (cache_dir={cache_dir})")
    ds = load_dataset(
        "openslr/librispeech_asr",
        split=split,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    print(f"[librispeech] {len(ds)} samples")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out = manifest_path.open("w", encoding="utf-8")
    n = 0
    for sample in tqdm(ds, unit="sample"):
        audio_path = sample["audio"]["path"]
        if not audio_path or not Path(audio_path).exists():
            continue
        text = sample.get("text") or sample.get("sentence") or ""
        if not text.strip():
            continue
        info = sample["audio"]
        try:
            duration = float(len(info["array"])) / float(info["sampling_rate"])
        except Exception:
            duration = -1.0
        rec = {
            "audio": str(Path(audio_path).resolve()),
            "text": text.strip(),
            "duration": round(duration, 3),
            "source": "english",
            "dataset": "librispeech",
            "split": split,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if max_items and n >= max_items:
            break
    out.close()
    print(f"[done] wrote {n} samples -> {manifest_path}")


def dump_common_voice(lang: str, split: str, manifest_path: Path, cache_dir: Path, max_items: int) -> None:
    from datasets import load_dataset

    print(f"[common_voice] {lang} split={split}")
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        lang,
        split=split,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    print(f"[common_voice] {len(ds)} samples")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out = manifest_path.open("w", encoding="utf-8")
    n = 0
    for sample in tqdm(ds, unit="sample"):
        audio_path = sample["audio"]["path"]
        text = sample.get("sentence") or ""
        if not text.strip() or not audio_path:
            continue
        try:
            info = sample["audio"]
            duration = float(len(info["array"])) / float(info["sampling_rate"])
        except Exception:
            duration = -1.0
        rec = {
            "audio": str(Path(audio_path).resolve()),
            "text": text.strip(),
            "duration": round(duration, 3),
            "source": "english",
            "dataset": "common_voice_17",
            "split": split,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if max_items and n >= max_items:
            break
    out.close()
    print(f"[done] wrote {n} samples -> {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=["librispeech", "common_voice"],
        default="librispeech",
    )
    ap.add_argument(
        "--split",
        default="train.clean.100",
        help="librispeech: clean.dev | clean.100 | clean.360 | other.500 | clean.test ; common_voice: train | validation | test",
    )
    ap.add_argument("--lang", default="en", help="ngôn ngữ (common_voice)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--max-items", type=int, default=0, help="0 = không giới hạn")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "librispeech":
        dump_librispeech(args.split, args.manifest, args.cache_dir, args.max_items)
    elif args.dataset == "common_voice":
        dump_common_voice(args.lang, args.split, args.manifest, args.cache_dir, args.max_items)


if __name__ == "__main__":
    main()
