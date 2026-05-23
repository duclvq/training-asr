"""Build manifest từ LibriSpeech extracted directory (downloaded từ openslr.org).

Cấu trúc LibriSpeech:
    LibriSpeech/<split>/<spk>/<chapter>/<spk>-<chapter>-<utt>.flac
    LibriSpeech/<split>/<spk>/<chapter>/<spk>-<chapter>.trans.txt

Output JSONL: 1 record/dòng, {"audio", "text", "duration", "source", "dataset", "split"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="path tới thư mục LibriSpeech/<split>/")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    out = args.manifest.open("w", encoding="utf-8")
    n = 0
    flac_files = list(args.root.rglob("*.flac"))
    print(f"found {len(flac_files)} flac files")

    # Build a lookup from utt_id -> text
    text_lookup: dict[str, str] = {}
    for tt in args.root.rglob("*.trans.txt"):
        with tt.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    text_lookup[parts[0]] = parts[1]
    print(f"loaded {len(text_lookup)} transcripts")

    for flac in tqdm(flac_files, unit="file"):
        utt_id = flac.stem
        text = text_lookup.get(utt_id)
        if not text:
            continue
        try:
            info = sf.info(str(flac))
            duration = info.frames / info.samplerate
        except Exception:
            continue
        rec = {
            "audio": str(flac.resolve()),
            "text": text.strip(),
            "duration": round(duration, 3),
            "source": "english",
            "dataset": "librispeech",
            "split": args.split,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if args.max_items and n >= args.max_items:
            break
    out.close()
    print(f"wrote {n} samples -> {args.manifest}")


if __name__ == "__main__":
    main()
