# RESULTS — Overnight ASR training run

Generated automatically. See `OVERNIGHT_LOG.md` for timeline.

## 1. Data

### VN-augmented (synthesized via F5-TTS)
- Count: **299** samples
- Total audio: 0.44h (1600.1s)
- Voices used: 15
- By category: {'province': 153, 'top_leader': 146}

### English (LibriSpeech)
- Count: **2703** samples, duration 5.39h

### Splits
- train: 1979
- eval: 137
- test_vn: 23
- test_en: 160

## 2. Evaluation results

| Run | Manifest | WER overall | CER overall | VN-name accuracy | n samples |
|---|---|---|---|---|---|
| `baseline_whisper_turbo_en` | `test_en.jsonl` | 3.34% | 1.61% | 0.0% (0) | — |
| `baseline_whisper_turbo_vn` | `test_vn.jsonl` | 2.07 | 1.40 | 0.0% (23) | — |

### Detail — `baseline_whisper_turbo_en`
- Model: `openai/whisper-large-v3-turbo` (type=whisper)
- Per-source breakdown:
  - `english` (n=160): WER=3.34% CER=1.61%

### Detail — `baseline_whisper_turbo_vn`
- Model: `openai/whisper-large-v3-turbo` (type=whisper)
- Per-source breakdown:
  - `vn_augmented` (n=23): WER=2.07 CER=1.40
- Top 10 VN names by samples:
  - `Tô Lâm`: 0/4 = 0.0%
  - `Lê Minh Hưng`: 0/3 = 0.0%
  - `Hà Nội`: 0/2 = 0.0%
  - `Trần Thanh Mẫn`: 0/2 = 0.0%
  - `Điện Biên`: 0/2 = 0.0%
  - `Phú Thọ`: 0/1 = 0.0%
  - `Lào Cai`: 0/1 = 0.0%
  - `Cà Mau`: 0/1 = 0.0%
  - `Huế`: 0/1 = 0.0%
  - `Lạng Sơn`: 0/1 = 0.0%

## 3. Files

- Manifests: `data/manifests/`
- Synthesized audio: `data/audio/synth/`
- Checkpoints: `checkpoints/`
- Predictions JSONL + summaries: `preds/`
