"""Đánh giá model ASR (Wav2Vec2 hoặc Whisper / Whisper+LoRA) trên 1 manifest.

Output:
- WER, CER overall
- WER/CER split theo `source` (vn_augmented vs english)
- "VN-name accuracy": % câu mà MỌI tên VN trong groundtruth xuất hiện đúng nguyên văn
  (giữ dấu) trong prediction. Đây là metric quan trọng cho mục tiêu đề tài.
- Lưu predictions JSONL để inspect lỗi.

Cách chạy:
    python scripts/evaluate_model.py \
        --manifest data/manifests/test_vn.jsonl \
        --model-type whisper \
        --model openai/whisper-large-v3-turbo \
        --lora-path checkpoints/whisper-turbo-lora/lora_final \
        --out preds_whisper_lora_vn.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
from tqdm import tqdm

import evaluate


def make_normalizer():
    """Use Whisper's BasicTextNormalizer (preserves diacritics) + lower."""
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        norm = BasicTextNormalizer()
        return lambda s: norm(s).strip()
    except Exception:
        # Fallback: lower + strip punctuation
        import re
        _re = re.compile(r"[\.\,\?\!\-\;\:\"\(\)\[\]\{\}']")
        return lambda s: _re.sub(" ", s.lower()).strip()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_audio(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy(), target_sr


def run_whisper(args, items):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval().to(args.device)

    forced_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    preds = []
    for it in tqdm(items, desc="whisper"):
        audio, sr = load_audio(it["audio"])
        feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(args.device)
        with torch.no_grad():
            ids = model.generate(feats, forced_decoder_ids=forced_ids, max_new_tokens=225)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        preds.append(text)
    return preds


def run_wav2vec2(args, items):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval().to(args.device)
    preds = []
    for it in tqdm(items, desc="wav2vec2"):
        audio, sr = load_audio(it["audio"])
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(args.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0].strip()
        preds.append(text)
    return preds


def vn_name_accuracy(preds, items):
    """% câu mà mọi vietnamese_term có trong prediction (case-sensitive, diacritics-preserving)."""
    ok = 0
    total = 0
    per_term = defaultdict(lambda: [0, 0])  # [hits, total]
    for p, it in zip(preds, items):
        terms = it.get("vietnamese_terms") or []
        if not terms:
            continue
        total += 1
        all_in = True
        for t in terms:
            per_term[t][1] += 1
            if t in p:
                per_term[t][0] += 1
            else:
                all_in = False
        if all_in:
            ok += 1
    return {
        "n_samples": total,
        "all_terms_present_pct": (ok / total * 100.0) if total else 0.0,
        "per_term": {t: {"hits": h, "total": tt, "pct": h / tt * 100 if tt else 0.0}
                     for t, (h, tt) in per_term.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-type", choices=["whisper", "wav2vec2"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    print(f"[eval] {len(items)} samples from {args.manifest}")

    if args.model_type == "whisper":
        preds = run_whisper(args, items)
    else:
        preds = run_wav2vec2(args, items)

    refs = [it["text"].strip() for it in items]
    normalizer = make_normalizer()
    norm_refs = [normalizer(r) for r in refs]
    norm_preds = [normalizer(p) for p in preds]
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    overall = {
        # Normalized WER/CER — fair comparison ignoring case/punct
        "wer": wer.compute(predictions=norm_preds, references=norm_refs),
        "cer": cer.compute(predictions=norm_preds, references=norm_refs),
        # Raw (no normalization) for transparency
        "wer_raw": wer.compute(predictions=preds, references=refs),
        "cer_raw": cer.compute(predictions=preds, references=refs),
    }

    by_src = defaultdict(lambda: {"preds": [], "refs": []})
    for p, it in zip(norm_preds, items):
        by_src[it.get("source", "?")]["preds"].append(p)
        by_src[it.get("source", "?")]["refs"].append(normalizer(it["text"].strip()))
    per_source = {
        src: {
            "n": len(v["preds"]),
            "wer": wer.compute(predictions=v["preds"], references=v["refs"]),
            "cer": cer.compute(predictions=v["preds"], references=v["refs"]),
        }
        for src, v in by_src.items()
    }
    # VN-name accuracy: count diacritic-preserving exact-match on ORIGINAL preds
    vn_acc = vn_name_accuracy(preds, items)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p, it in zip(preds, items):
            f.write(json.dumps({
                "id": Path(it["audio"]).stem,
                "ref": it["text"],
                "hyp": p,
                "source": it.get("source"),
                "vietnamese_terms": it.get("vietnamese_terms", []),
            }, ensure_ascii=False) + "\n")

    summary = {
        "manifest": str(args.manifest),
        "model_type": args.model_type,
        "model": args.model,
        "lora_path": args.lora_path,
        "overall": overall,
        "per_source": per_source,
        "vn_name_accuracy": vn_acc,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Print to stderr to avoid Windows console encoding issues with VN diacritics
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)
    print(f"[done] preds -> {args.out}  summary -> {summary_path}")


if __name__ == "__main__":
    main()
