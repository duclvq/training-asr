"""Tổng hợp kết quả overnight run thành RESULTS.md.

Đọc các file summary từ scripts/evaluate_model.py + manifest stats + log,
dán vào template để user xem nhanh sáng mai.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def safe_load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": str(e)}


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for _ in p.open(encoding="utf-8"))


def manifest_stats(p: Path) -> dict:
    if not p.exists():
        return {"present": False}
    n = 0
    dur = 0.0
    sources = Counter()
    cats = Counter()
    voices = Counter()
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            d = r.get("duration") or 0
            if d > 0:
                dur += d
            sources[r.get("source", "?")] += 1
            cats[r.get("category", "?")] += 1
            if r.get("ref_voice"):
                voices[r["ref_voice"]] += 1
    return {
        "present": True,
        "count": n,
        "total_duration_sec": round(dur, 1),
        "total_duration_hr": round(dur / 3600, 2),
        "by_source": dict(sources),
        "by_category": dict(cats),
        "n_voices": len(voices),
        "top_voices": voices.most_common(5),
    }


def format_metric(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v*100:.2f}%" if v <= 1 else f"{v:.2f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BASE_DIR / "RESULTS.md")
    args = ap.parse_args()

    # Manifests
    synth = manifest_stats(BASE_DIR / "data" / "manifests" / "synth_manifest.jsonl")
    eng = manifest_stats(BASE_DIR / "data" / "manifests" / "english_manifest.jsonl")
    train = count_lines(BASE_DIR / "data" / "manifests" / "train.jsonl")
    eval_n = count_lines(BASE_DIR / "data" / "manifests" / "eval.jsonl")
    test_vn = count_lines(BASE_DIR / "data" / "manifests" / "test_vn.jsonl")
    test_en = count_lines(BASE_DIR / "data" / "manifests" / "test_en.jsonl")

    # Eval summaries
    pred_dir = BASE_DIR / "preds"
    summaries: dict[str, dict] = {}
    if pred_dir.exists():
        for s in sorted(pred_dir.glob("*.summary.json")):
            summaries[s.stem.replace(".summary", "")] = json.loads(s.read_text(encoding="utf-8"))

    # Build markdown
    lines = []
    lines.append("# RESULTS — Overnight ASR training run")
    lines.append("")
    lines.append("Generated automatically. See `OVERNIGHT_LOG.md` for timeline.")
    lines.append("")
    lines.append("## 1. Data")
    lines.append("")
    lines.append("### VN-augmented (synthesized via F5-TTS)")
    if synth.get("present"):
        lines.append(f"- Count: **{synth['count']}** samples")
        lines.append(f"- Total audio: {synth['total_duration_hr']}h ({synth['total_duration_sec']}s)")
        lines.append(f"- Voices used: {synth['n_voices']}")
        lines.append(f"- By category: {synth['by_category']}")
    else:
        lines.append("- (manifest not present)")
    lines.append("")
    lines.append("### English (LibriSpeech)")
    if eng.get("present"):
        lines.append(f"- Count: **{eng['count']}** samples, duration {eng['total_duration_hr']}h")
    else:
        lines.append("- (manifest not present)")
    lines.append("")
    lines.append("### Splits")
    lines.append(f"- train: {train}")
    lines.append(f"- eval: {eval_n}")
    lines.append(f"- test_vn: {test_vn}")
    lines.append(f"- test_en: {test_en}")
    lines.append("")

    lines.append("## 2. Evaluation results")
    lines.append("")
    if not summaries:
        lines.append("_No eval summaries yet._")
    else:
        lines.append("| Run | Manifest | WER overall | CER overall | VN-name accuracy | n samples |")
        lines.append("|---|---|---|---|---|---|")
        for name, s in summaries.items():
            ov = s.get("overall", {})
            vn = s.get("vn_name_accuracy", {})
            lines.append(
                f"| `{name}` | `{Path(s.get('manifest','')).name}` "
                f"| {format_metric(ov.get('wer'))} "
                f"| {format_metric(ov.get('cer'))} "
                f"| {vn.get('all_terms_present_pct', 0):.1f}% ({vn.get('n_samples', 0)}) "
                f"| — |"
            )
        lines.append("")

        # Detail per run
        for name, s in summaries.items():
            lines.append(f"### Detail — `{name}`")
            lines.append(f"- Model: `{s.get('model')}` (type={s.get('model_type')})")
            if s.get("lora_path"):
                lines.append(f"- LoRA: `{s.get('lora_path')}`")
            ps = s.get("per_source", {})
            if ps:
                lines.append("- Per-source breakdown:")
                for src, m in ps.items():
                    lines.append(
                        f"  - `{src}` (n={m['n']}): WER={format_metric(m['wer'])} CER={format_metric(m['cer'])}"
                    )
            vn = s.get("vn_name_accuracy", {})
            per_term = vn.get("per_term", {})
            if per_term:
                lines.append("- Top 10 VN names by samples:")
                top = sorted(per_term.items(), key=lambda x: -x[1]["total"])[:10]
                for term, t in top:
                    lines.append(f"  - `{term}`: {t['hits']}/{t['total']} = {t['pct']:.1f}%")
            lines.append("")

    lines.append("## 3. Files")
    lines.append("")
    lines.append("- Manifests: `data/manifests/`")
    lines.append("- Synthesized audio: `data/audio/synth/`")
    lines.append("- Checkpoints: `checkpoints/`")
    lines.append("- Predictions JSONL + summaries: `preds/`")
    lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
