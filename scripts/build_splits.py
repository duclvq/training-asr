"""Trộn manifest VN-augmented + English, chia thành train/eval/test.

Output:
    data/manifests/train.jsonl
    data/manifests/eval.jsonl
    data/manifests/test.jsonl

Test set giữ riêng `vn_augmented` để đánh giá khả năng nhận diện tên VN.

Cách chạy:
    python scripts/build_splits.py \
        --vn-manifest data/manifests/synth_manifest.jsonl \
        --en-manifest data/manifests/english_manifest.jsonl \
        --en-sample 5000 \
        --eval-ratio 0.05 --test-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE_DIR / "data" / "manifests"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vn-manifest", type=Path, default=MANIFEST_DIR / "synth_manifest.jsonl")
    ap.add_argument("--en-manifest", type=Path, default=MANIFEST_DIR / "english_manifest.jsonl")
    ap.add_argument("--en-sample", type=int, default=0, help="0 = lấy hết English manifest")
    ap.add_argument("--eval-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=MANIFEST_DIR)
    args = ap.parse_args()

    random.seed(args.seed)

    vn = read_jsonl(args.vn_manifest)
    en = read_jsonl(args.en_manifest)
    print(f"[in] vn={len(vn)}  en={len(en)}")
    if args.en_sample and len(en) > args.en_sample:
        en = random.sample(en, args.en_sample)
        print(f"[sample] en -> {len(en)}")

    def split(items, eval_r, test_r):
        random.shuffle(items)
        n = len(items)
        n_eval = max(1, int(n * eval_r)) if n else 0
        n_test = max(1, int(n * test_r)) if n else 0
        eval_set = items[:n_eval]
        test_set = items[n_eval:n_eval + n_test]
        train_set = items[n_eval + n_test:]
        return train_set, eval_set, test_set

    # VN-augmented: split nguyên-vẹn để cả 3 set đều có audio chứa tên VN
    vn_train, vn_eval, vn_test = split(vn, args.eval_ratio, args.test_ratio)
    en_train, en_eval, en_test = split(en, args.eval_ratio, args.test_ratio)

    train = vn_train + en_train
    eval_set = vn_eval + en_eval
    # Test riêng theo source để báo cáo metric riêng cho VN names
    test_vn = vn_test
    test_en = en_test

    random.shuffle(train)
    random.shuffle(eval_set)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "eval.jsonl", eval_set)
    write_jsonl(args.out_dir / "test_vn.jsonl", test_vn)
    write_jsonl(args.out_dir / "test_en.jsonl", test_en)

    print(
        f"[out] train={len(train)}  eval={len(eval_set)}  "
        f"test_vn={len(test_vn)}  test_en={len(test_en)}"
    )


if __name__ == "__main__":
    main()
