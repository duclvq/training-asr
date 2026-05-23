"""LoRA fine-tune Whisper Turbo (openai/whisper-large-v3-turbo).

LoRA target: q_proj, k_proj, v_proj, out_proj của attention layers.
Language: tiếng Anh (en) — task: transcribe.

Input: 2 manifest JSONL (train, eval), mỗi dòng:
    {"audio": "/abs/path.wav", "text": "...", "duration": <s>, "source": "..."}

Cách chạy:
    python scripts/finetune_whisper_lora.py \
        --train-manifest data/manifests/train.jsonl \
        --eval-manifest  data/manifests/eval.jsonl \
        --output-dir checkpoints/whisper-turbo-lora

GPU yêu cầu: ~16-20GB VRAM với batch 4 + LoRA (turbo có ~800M params).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import librosa
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

import evaluate


CHARS_TO_NORM = re.compile(r"\s+")


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
    return CHARS_TO_NORM.sub(" ", text).strip()


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


class ASRDataset(Dataset):
    """Torch Dataset — load audio on the fly, no datasets.map / multiprocess pipes."""
    def __init__(self, items, feature_extractor, tokenizer, sample_rate: int, max_duration: float):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        # Pre-filter by duration
        self.items = [
            it for it in items
            if (it.get("duration", -1.0) <= 0 or it.get("duration", 0) <= max_duration)
            and normalize_text(it.get("text") or "")
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        audio = _load_audio(it["audio"], self.sample_rate)
        feats = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features[0]
        labels = self.tokenizer(normalize_text(it["text"])).input_ids
        return {"input_features": feats, "labels": labels}


@dataclass
class DataCollatorWhisper:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--eval-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-duration", type=float, default=30.0)  # Whisper trần 30s
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)  # LoRA lr lớn hơn full-ft
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,out_proj,fc1,fc2",
                    help="comma-sep tên module để LoRA wrap")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)

    train_items = load_manifest(args.train_manifest)
    eval_items = load_manifest(args.eval_manifest)
    print(f"[data] train={len(train_items)}  eval={len(eval_items)}")

    train_ds = ASRDataset(train_items, feature_extractor, tokenizer, args.sample_rate, args.max_duration)
    eval_ds = ASRDataset(eval_items, feature_extractor, tokenizer, args.sample_rate, args.max_duration)
    print(f"[ds] train={len(train_ds)}  eval={len(eval_ds)} (sau filter duration)")

    model_kwargs: dict[str, Any] = {}
    if args.load_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.load_8bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,  # avoid Windows multiprocess pipe issue
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWhisper(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    model.save_pretrained(str(args.output_dir / "lora_final"))
    processor.save_pretrained(str(args.output_dir / "lora_final"))
    metrics = trainer.evaluate()
    (args.output_dir / "lora_final" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
