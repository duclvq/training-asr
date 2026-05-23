# Kế hoạch huấn luyện · thử nghiệm · đánh giá

Mục tiêu: huấn luyện ASR tiếng Anh **nhận diện chính xác tên riêng tiếng Việt** (lãnh đạo, tỉnh, địa danh) trong câu English có giọng/phát âm kiểu người Việt. So sánh 2 hướng:

1. **Wav2Vec2 full-finetune** với vocab mở rộng cho ký tự VN.
2. **Whisper Turbo + LoRA** (giữ encoder gốc, chỉ học thêm adapter).

Baseline để so sánh: **Whisper Turbo zero-shot** (không finetune).

---

## Giai đoạn 1 — Data (1–2 ngày)

### 1.1. Sinh text
```powershell
# đã chạy: 1000 câu province + top_leader (job bp0mqv17k)
# Sau này mở rộng:
python generate_dataset.py --count 2000 --pools province,top_leader,leader,landmark,mixed --batch-size 25
```
Output: `data/generated_sentences.jsonl`

### 1.2. Chuẩn bị reference audio cho TTS
Cần 1–3 file audio mẫu giọng tiếng Việt (3–10s, sạch, không nhạc nền) + transcript đúng nguyên văn.
- Lý tưởng: nhiều giọng khác nhau (nam/nữ, Bắc/Trung/Nam) → đa dạng âm sắc trong dataset.
- Lưu tại `data/refs/<voice_id>.wav` + `data/refs/<voice_id>.txt`.

### 1.3. Synthesize audio từ jsonl
```powershell
# 1 giọng:
python scripts/synthesize_audio.py `
  --ref-audio data/refs/voice1.wav `
  --ref-text  "Nội dung verbatim của voice1.wav"

# Nhiều giọng: chạy nhiều lần với --out-dir và --manifest khác nhau, sau đó concat manifest.
```
Output: `data/audio/synth/<id>.wav` + `data/manifests/synth_manifest.jsonl`.

Thời lượng ước tính: ~10s/câu trên GPU → 1000 câu ~ 3h. Mỗi câu ~5–8s audio → ~1h–2h speech.

### 1.4. Tải English dataset
```powershell
# nhanh để smoke-test (~340MB, 5h):
python scripts/download_english.py --dataset librispeech --split clean.dev

# scale-up (~6GB, 100h):
python scripts/download_english.py --dataset librispeech --split train.clean.100

# alternative: Common Voice (giọng đa dạng hơn):
python scripts/download_english.py --dataset common_voice --lang en --split train --max-items 20000
```
Output: `data/manifests/english_manifest.jsonl`.

### 1.5. Tạo splits
```powershell
python scripts/build_splits.py --en-sample 5000 --eval-ratio 0.05 --test-ratio 0.05
```
Output: `train.jsonl`, `eval.jsonl`, `test_vn.jsonl`, `test_en.jsonl`.

Tỷ lệ trộn khuyến nghị (phải tinh chỉnh):
- Smoke run: 1000 VN + 5000 EN
- Production: 5000–10000 VN + 50000+ EN

---

## Giai đoạn 2 — Huấn luyện (1–3 ngày tùy GPU)

### 2.1. Whisper Turbo + LoRA (nên chạy trước — nhẹ, nhanh hội tụ)
```powershell
python scripts/finetune_whisper_lora.py `
  --train-manifest data/manifests/train.jsonl `
  --eval-manifest  data/manifests/eval.jsonl `
  --output-dir checkpoints/whisper-turbo-lora `
  --epochs 5 --batch-size 4 --grad-accum 4 --bf16
```
Yêu cầu VRAM: ~16–20GB. Nếu chỉ có 12GB: thêm `--load-8bit`. Nếu 8GB: giảm batch=2, grad-accum=8.

Tiên đoán: WER trên `test_vn` giảm 30–60% so với zero-shot baseline, VN-name accuracy tăng đáng kể.

### 2.2. Wav2Vec2 full finetune
```powershell
python scripts/finetune_wav2vec2.py `
  --train-manifest data/manifests/train.jsonl `
  --eval-manifest  data/manifests/eval.jsonl `
  --output-dir checkpoints/wav2vec2-vnaug `
  --epochs 15 --batch-size 8 --grad-accum 2 --bf16
```
Yêu cầu VRAM: ~16GB (large) hoặc 8GB (base).

Lưu ý:
- Vocab CTC build từ training transcripts → nếu thiếu ký tự VN nào, dataset chưa đủ.
- Wav2Vec2 base ASR có thể converge chậm vì re-init head — cần ≥10 epoch hoặc training data lớn.

### 2.3. Baseline (cần làm để so sánh)
```powershell
python scripts/evaluate_model.py `
  --manifest data/manifests/test_vn.jsonl `
  --model-type whisper --model openai/whisper-large-v3-turbo `
  --out preds/baseline_whisper_turbo_vn.jsonl
```

---

## Giai đoạn 3 — Đánh giá (½ ngày)

### 3.1. Metric chính
| Metric | Ý nghĩa | Quan trọng |
|---|---|---|
| **WER** overall | Word Error Rate toàn bộ | Tham khảo |
| **WER** trên `test_vn` | Bao gồm cả từ tiếng Anh + tên VN | Cao |
| **CER** trên `test_vn` | Char-level — phát hiện mất/nhầm dấu | Cao |
| **VN-name accuracy** | % câu mà MỌI tên VN trong gt xuất hiện đúng nguyên văn trong hyp | **Cao nhất** |
| **WER** trên `test_en` | Để chắc chắn không "quên" tiếng Anh gốc (catastrophic forgetting) | Cao |

### 3.2. Chạy đánh giá toàn diện
```powershell
# Baseline:
python scripts/evaluate_model.py --manifest data/manifests/test_vn.jsonl --model-type whisper `
  --model openai/whisper-large-v3-turbo --out preds/baseline_vn.jsonl
python scripts/evaluate_model.py --manifest data/manifests/test_en.jsonl --model-type whisper `
  --model openai/whisper-large-v3-turbo --out preds/baseline_en.jsonl

# Whisper LoRA:
python scripts/evaluate_model.py --manifest data/manifests/test_vn.jsonl --model-type whisper `
  --model openai/whisper-large-v3-turbo --lora-path checkpoints/whisper-turbo-lora/lora_final `
  --out preds/whisper_lora_vn.jsonl
python scripts/evaluate_model.py --manifest data/manifests/test_en.jsonl --model-type whisper `
  --model openai/whisper-large-v3-turbo --lora-path checkpoints/whisper-turbo-lora/lora_final `
  --out preds/whisper_lora_en.jsonl

# Wav2Vec2:
python scripts/evaluate_model.py --manifest data/manifests/test_vn.jsonl --model-type wav2vec2 `
  --model checkpoints/wav2vec2-vnaug/final --out preds/w2v2_vn.jsonl
```

### 3.3. Inspection thủ công
- Đọc 30–50 sample lỗi worst-case trong `preds/*.jsonl`.
- Phân loại lỗi:
  - Mất dấu (Hà → Ha)
  - Nhầm ký tự (Hạ Long → Ha Lon)
  - Nhầm tên (Phú Thọ → Phu Tho hay Phú Quốc?)
  - Bỏ qua tên hoàn toàn

---

## Giai đoạn 4 — Ablation / mở rộng (1–2 tuần)

Khi pipeline base chạy được, các thí nghiệm nên làm theo độ ưu tiên:

### Ưu tiên cao
1. **Ablation tỷ lệ VN:EN** — thử 1:1, 1:5, 1:10, 1:0 (chỉ VN).
   Hypothesis: VN ratio quá cao → quên tiếng Anh; quá thấp → không học được tên VN.
2. **Đa giọng reference** — tổng hợp với 3–5 giọng khác nhau thay vì 1 giọng.
   Hypothesis: tăng generalization.
3. **LoRA r/alpha sweep** — r ∈ {8, 16, 32, 64}.

### Ưu tiên trung bình
4. **Tăng dataset VN** lên 5k–10k câu.
5. **Augmentation audio** — SpecAugment, noise injection (xem `data/noises/`).
6. **Whisper Large v3 (không turbo)** — so sánh nếu VRAM cho phép.

### Ưu tiên thấp / sau cùng
7. **Wav2Vec2 XLS-R 300M/1B** thay vì base — multilingual encoder mạnh hơn cho VN sounds.
8. **Beam search + LM** (KenLM) cho Wav2Vec2.
9. **Chuyển hết toàn bộ leader/landmark vào pool** (mở rộng `--pools`).

---

## Rủi ro & cách giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| F5-TTS đọc sai tên VN (artifact, nuốt âm) | Inspect random 50 mẫu wav trước khi train. Loại bỏ bằng tay nếu cần. Có thể tăng `--remove-silence`. |
| Catastrophic forgetting tiếng Anh | LoRA mặc định nhẹ → ít forget. Wav2Vec2 cần giữ EN trong train mix. Đánh giá luôn cả `test_en`. |
| Vocab CTC thiếu ký tự VN | Generate vocab từ toàn bộ transcript, kiểm tra log "[vocab] N symbols" — phải có ă, â, ê, ô, ơ, ư, đ + dấu thanh. |
| GPU OOM | `--bf16`, `gradient_checkpointing` (đã bật), giảm batch, `--load-8bit` cho Whisper. |
| Eval data leak | `build_splits.py` shuffle rồi cắt; không có duplicate cross-split. Có thể audit bằng `id` của câu sinh. |

---

## Tiến độ checklist

- [x] Sinh data text (1000 câu province + top_leader)
- [ ] Reference audio (≥1 giọng)
- [ ] Synthesize audio
- [ ] Tải English data
- [ ] Splits
- [ ] Baseline eval (Whisper zero-shot)
- [ ] Whisper LoRA train
- [ ] Whisper LoRA eval
- [ ] Wav2Vec2 train
- [ ] Wav2Vec2 eval
- [ ] Báo cáo so sánh + inspection
