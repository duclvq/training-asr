# Script chạy sau khi reboot để hoàn tất training + eval + report.
# Cách dùng:
#   PowerShell> cd E:\py_source\training_asr
#   PowerShell> .\scripts\continue_pipeline.ps1
#
# Hoặc với cờ -SkipTrain để chỉ chạy eval + report (nếu đã train trước đó).

param(
    [switch]$SkipTrain,
    [switch]$AlsoTrainWav2Vec2,
    [string]$Precision = "bf16"   # hoặc "fp16"
)

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host "=== 0. Kiểm tra GPU ==="
nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Error "GPU không sẵn sàng. Kiểm tra driver hoặc reboot lại."
}

if (-not $SkipTrain) {
    Write-Host "`n=== 1. Train Whisper Turbo + LoRA ==="
    $precFlag = "--$Precision"
    python scripts/finetune_whisper_lora.py `
        --train-manifest data/manifests/train.jsonl `
        --eval-manifest  data/manifests/eval.jsonl `
        --output-dir checkpoints/whisper-turbo-lora `
        --epochs 5 --batch-size 4 --grad-accum 4 $precFlag `
        --lora-r 32 --lora-alpha 64 --warmup-steps 50
    if ($LASTEXITCODE -ne 0) { Write-Error "Whisper LoRA train failed" }
}

Write-Host "`n=== 2. Eval Whisper+LoRA trên test_vn ==="
python scripts/evaluate_model.py `
    --manifest data/manifests/test_vn.jsonl `
    --model-type whisper --model openai/whisper-large-v3-turbo `
    --lora-path checkpoints/whisper-turbo-lora/lora_final `
    --out preds/whisper_lora_vn.jsonl
if ($LASTEXITCODE -ne 0) { Write-Error "Eval whisper_lora_vn failed" }

Write-Host "`n=== 3. Eval Whisper+LoRA trên test_en ==="
python scripts/evaluate_model.py `
    --manifest data/manifests/test_en.jsonl `
    --model-type whisper --model openai/whisper-large-v3-turbo `
    --lora-path checkpoints/whisper-turbo-lora/lora_final `
    --out preds/whisper_lora_en.jsonl
if ($LASTEXITCODE -ne 0) { Write-Error "Eval whisper_lora_en failed" }

if ($AlsoTrainWav2Vec2) {
    Write-Host "`n=== 4. Train Wav2Vec2 (tuỳ chọn) ==="
    python scripts/finetune_wav2vec2.py `
        --train-manifest data/manifests/train.jsonl `
        --eval-manifest  data/manifests/eval.jsonl `
        --output-dir checkpoints/wav2vec2-vnaug `
        --epochs 15 --batch-size 8 --grad-accum 2 --$Precision

    Write-Host "`n=== 5. Eval Wav2Vec2 ==="
    python scripts/evaluate_model.py `
        --manifest data/manifests/test_vn.jsonl `
        --model-type wav2vec2 --model checkpoints/wav2vec2-vnaug/final `
        --out preds/w2v2_vn.jsonl
    python scripts/evaluate_model.py `
        --manifest data/manifests/test_en.jsonl `
        --model-type wav2vec2 --model checkpoints/wav2vec2-vnaug/final `
        --out preds/w2v2_en.jsonl
}

Write-Host "`n=== 6. Compile RESULTS.md ==="
python scripts/compile_report.py

Write-Host "`n✅ Done! Xem RESULTS.md"
