"""Fine-tune Wav2Vec2 CTC trên hỗn hợp manifest VN-augmented + English.

Khác biệt với recipe HF chuẩn:
1. Build vocab tự động từ TẤT CẢ transcript trong cả 2 manifest -> để vocab phủ ký tự VN
   có dấu (ă, â, đ, ê, ô, ơ, ư + dấu thanh).
2. Khởi tạo head CTC mới có size = len(vocab); freeze feature extractor mặc định
   (giống recipe HF), unfreeze nếu --unfreeze-features.
3. Backbone mặc định: facebook/wav2vec2-large-960h-lv60-self. Có thể đổi qua --model.

Cách chạy:
    python scripts/finetune_wav2vec2.py \
        --train-manifest data/manifests/train.jsonl \
        --eval-manifest  data/manifests/eval.jsonl \
        --output-dir checkpoints/wav2vec2-vnaug

Mặc định resample 16kHz, max 20s/sample.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import librosa
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

import evaluate

CHARS_TO_REMOVE = re.compile(r"[\,\?\.\!\-\;\:\"\(\)\[\]\{\}]")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_text(text: str) -> str:
    text = text.strip()
    text = CHARS_TO_REMOVE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_vocab(transcripts: list[str]) -> dict[str, int]:
    """Vocab = tập ký tự xuất hiện (lowercased) + ký tự đặc biệt."""
    chars = set()
    for t in transcripts:
        chars.update(t.lower())
    chars.discard(" ")
    vocab_list = sorted(chars)
    vocab = {c: i for i, c in enumerate(vocab_list)}
    vocab["|"] = len(vocab)  # word boundary
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def build_dataset(items: list[dict[str, Any]], sample_rate: int, max_duration: float) -> Dataset:
    rows = []
    for it in items:
        d = it.get("duration", -1.0)
        if 0 < d > max_duration:
            continue
        if d == 0:
            continue
        text = normalize_text(it["text"]).lower()
        if not text:
            continue
        rows.append({
            "audio_path": it["audio"],
            "text": text,
            "source": it.get("source", "unknown"),
        })
    return Dataset.from_list(rows)


def prepare_features(batch, processor, sample_rate: int):
    audio = _load_audio(batch["audio_path"], sample_rate)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="np")
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


@dataclass
class DataCollatorCTC:
    processor: Any
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--eval-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-large-960h-lv60-self")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--unfreeze-features", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_items = load_manifest(args.train_manifest)
    eval_items = load_manifest(args.eval_manifest)
    print(f"[data] train={len(train_items)}  eval={len(eval_items)}")

    all_texts = [normalize_text(it["text"]).lower() for it in train_items + eval_items]
    vocab = build_vocab(all_texts)
    vocab_path = args.output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"[vocab] {len(vocab)} symbols -> {vocab_path}")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=args.sample_rate, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(str(args.output_dir))

    train_ds = build_dataset(train_items, args.sample_rate, args.max_duration)
    eval_ds = build_dataset(eval_items, args.sample_rate, args.max_duration)
    print(f"[ds] train={len(train_ds)}  eval={len(eval_ds)} (sau filter duration)")

    train_ds = train_ds.map(lambda b: prepare_features(b, processor, args.sample_rate),
                            remove_columns=train_ds.column_names, num_proc=1, desc="prep train")
    eval_ds = eval_ds.map(lambda b: prepare_features(b, processor, args.sample_rate),
                          remove_columns=eval_ds.column_names, num_proc=1, desc="prep eval")

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    if not args.unfreeze_features:
        model.freeze_feature_encoder()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=400,
        save_total_limit=3,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,
        data_collator=DataCollatorCTC(processor=processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(str(args.output_dir / "final"))
    metrics = trainer.evaluate()
    (args.output_dir / "final" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
