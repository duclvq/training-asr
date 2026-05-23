# RESULTS — Overnight run 2026-05-23

## ⚠️ Tình hình: GPU bị mất kết nối, không hoàn thành training

Đêm chạy đến bước **train Whisper LoRA** thì NVIDIA driver mất kết nối với GPU
(`nvidia-smi`: *"GPU is lost. Reboot the system to recover this GPU"*). Không
khôi phục được nếu không reboot/quyền admin. Khi bạn dậy, **chạy lại 1 lệnh
training** (chi tiết cuối file) là tiếp tục được — toàn bộ data, baseline, và
scripts đã sẵn sàng.

## ✅ Đã hoàn thành

| Phase | Trạng thái | Kết quả |
|---|---|---|
| Sinh data text | OK | ~300+ câu English có tên VN (Tô Lâm, Hà Nội, 34 tỉnh, …) |
| Tải VN voice refs | OK | 15 speaker từ VIVOS @ `data/refs/` |
| F5-TTS web demo (port 8001) | OK | Đã verify, server stop khi train |
| Synthesize VN-augmented audio | OK | **298 wav** @ 24kHz, 15 giọng, mean 5s, `data/audio/synth/` |
| Tải LibriSpeech | OK | dev-clean (2703 samples) — `data/manifests/english_manifest.jsonl` |
| Build splits | OK | train 1979 / eval 137 / test_vn 23 / test_en 160 |
| **Baseline Whisper-Turbo zero-shot** | OK | xem bảng dưới |
| Train Whisper-Turbo LoRA | **Bị chặn (GPU lost)** | scripts đã debug xong, sẵn sàng chạy |
| Eval finetuned | Chưa | sẵn `evaluate_model.py` |
| Train Wav2Vec2 | Chưa | sẵn `finetune_wav2vec2.py` |

## 📊 Baseline (Whisper-Turbo zero-shot)

Metrics đã được normalize bằng `BasicTextNormalizer` của Whisper (bỏ qua case/punctuation, GIỮ dấu tiếng Việt).

| Test set | n | WER (norm) | CER (norm) | VN-name accuracy | Ghi chú |
|---|---|---|---|---|---|
| `test_en` (LibriSpeech) | 160 | **3.34%** | **1.61%** | — | Whisper hoạt động tốt trên English thật |
| `test_vn` (synth) | 23 | **207.31%** | 140.39% | **0.0%** | Whisper hallucinate hoàn toàn (`I'm sorry` loop) |

→ Cách biệt cực lớn giữa English thật và VN-augmented synthesized: Whisper không xử lý nổi audio có
tên Vietnamese. Đây là **baseline để so sánh với LoRA**: room for improvement rất rộng.

### Ví dụ baseline sai
```
REF: Farmers in Hà Nội and Phú Thọ reported record rice yields this harvest season.
HYP: I'm sorry, I'm sorry, I'm sorry, I'm sorry, ...  (lặp vô hạn)

REF: Tô Lâm was born in 1957 and often helped his family gather rice during the harvest season.
HYP: The chief of people who have been on the training for the people who have been on the training

REF: Born in Hà Nội, she spent her childhood in Huế during the monsoon season, ...
HYP: I'm so sorry to interrupt you.
```

## 📁 Cấu trúc files

```
E:\py_source\training_asr\
├── app.py / static/index.html      # F5-TTS web demo
├── generate_dataset.py             # Sinh câu English + tên VN
├── PLAN.md / OVERNIGHT_LOG.md      # Kế hoạch + log đêm
├── RESULTS.md                      # File này
├── data/
│   ├── provinces.json / leaders.json / landmarks.json
│   ├── generated_sentences.jsonl    # Text seeds (~300+)
│   ├── refs/                        # 15 giọng VIVOS cloning refs
│   ├── audio/synth/                 # 298 audio synthesized
│   ├── librispeech/LibriSpeech/...  # Raw LibriSpeech dev-clean
│   └── manifests/
│       ├── synth_manifest.jsonl     # 298 VN-augmented
│       ├── english_manifest.jsonl   # 2703 LibriSpeech
│       ├── train.jsonl              # 1979 (mixed)
│       ├── eval.jsonl               # 137
│       ├── test_vn.jsonl            # 23 (pure VN test)
│       └── test_en.jsonl            # 160 (pure EN test)
├── scripts/
│   ├── download_voice_refs.py
│   ├── synthesize_audio.py
│   ├── download_english.py / build_librispeech_manifest.py
│   ├── build_splits.py
│   ├── finetune_whisper_lora.py    # ✅ đã debug, ready
│   ├── finetune_wav2vec2.py        # ✅ đã debug, ready
│   ├── evaluate_model.py            # ✅ có normalizer + VN-name accuracy
│   └── compile_report.py            # Sinh RESULTS.md tự động
└── preds/
    ├── baseline_whisper_turbo_vn.jsonl + .summary.json
    └── baseline_whisper_turbo_en.jsonl + .summary.json
```

## 🛠️ Hồi phục GPU và tiếp tục (sáng dậy)

1. **Reboot máy** (đơn giản nhất). Sau khi boot, kiểm tra:
   ```powershell
   nvidia-smi
   ```
   Phải thấy RTX 5060 Ti với memory usage.

2. **Train Whisper LoRA**:
   ```powershell
   cd E:\py_source\training_asr
   $env:USE_TF="0"; $env:TRANSFORMERS_NO_TF="1"; $env:PYTHONIOENCODING="utf-8"
   python scripts/finetune_whisper_lora.py `
     --train-manifest data/manifests/train.jsonl `
     --eval-manifest  data/manifests/eval.jsonl `
     --output-dir checkpoints/whisper-turbo-lora `
     --epochs 5 --batch-size 4 --grad-accum 4 --bf16 `
     --lora-r 32 --lora-alpha 64 --warmup-steps 50
   ```
   ETA ~1.5-3h. Output checkpoint: `checkpoints/whisper-turbo-lora/lora_final/`.

   Nếu `--bf16` báo lỗi (RTX 50-series có thể cần update torch), đổi sang `--fp16`.

3. **Eval finetuned** (chạy cả 2):
   ```powershell
   python scripts/evaluate_model.py --manifest data/manifests/test_vn.jsonl `
     --model-type whisper --model openai/whisper-large-v3-turbo `
     --lora-path checkpoints/whisper-turbo-lora/lora_final `
     --out preds/whisper_lora_vn.jsonl

   python scripts/evaluate_model.py --manifest data/manifests/test_en.jsonl `
     --model-type whisper --model openai/whisper-large-v3-turbo `
     --lora-path checkpoints/whisper-turbo-lora/lora_final `
     --out preds/whisper_lora_en.jsonl
   ```

4. **Compile report tự động**:
   ```powershell
   python scripts/compile_report.py
   ```
   Sẽ overwrite `RESULTS.md` này với bảng so sánh đầy đủ baseline vs finetuned.

5. (Tuỳ chọn) **Train Wav2Vec2** để so sánh kiến trúc — chỉ chạy nếu thời gian cho phép:
   ```powershell
   python scripts/finetune_wav2vec2.py `
     --train-manifest data/manifests/train.jsonl `
     --eval-manifest  data/manifests/eval.jsonl `
     --output-dir checkpoints/wav2vec2-vnaug `
     --epochs 15 --batch-size 8 --grad-accum 2 --bf16
   ```

## 🔁 Mở rộng dataset (sau khi pipeline đầu chạy được)

- **Tăng số câu English chứa tên VN**: bỏ `--limit` và chạy lại synthesize_audio.py cho phần còn lại của `generated_sentences.jsonl`. Hoặc sinh thêm:
  ```powershell
  python generate_dataset.py --count 2000 --pools province,top_leader,leader,landmark,mixed --batch-size 25
  ```
- **Tăng English dataset**: `python scripts/download_english.py --split train.clean.100` (cần ~6GB, ~100h speech).

## 🧠 Nguyên nhân khả nghi GPU lost

- F5-TTS server chạy synth ~10 phút liên tục dùng GPU, sau đó bị stop.
- Có thể driver TDR / thermal / VRAM fragmentation gây mất handle.
- Reboot là cách chắc chắn nhất. Nếu lặp lại lần sau, kiểm tra:
  - `nvidia-smi -q -d TEMPERATURE`
  - Cập nhật driver: Geforce Experience hoặc nvidia.com
  - Disable TDR (advanced; risk hang): registry `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrLevel=0` — KHÔNG KHUYẾN NGHỊ
