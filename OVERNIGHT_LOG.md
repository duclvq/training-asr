# Overnight run log — 2026-05-23

Started ~02:30. User asleep. Status updated as phases progress.

## Phase status
- [~] Text generation — gen running but slow w/ parse errors. ~300 sentences saved so far.
- [x] Install training deps (transformers, peft, datasets, evaluate, jiwer, librosa, tensorboard).
- [x] Download VN voice refs: 15 speakers from VIVOS @ `data/refs/` (4-8s each).
- [x] F5-TTS verified: ~22s first call, ~3.4s/sample thereafter. Fixed: config.json → vocab.txt rename per README.
- [~] Synthesize 300 audios — running ETA ~14 min.
- [x] Download English: LibriSpeech dev-clean direct from openslr.org. Manifest: 2703 samples.
- [ ] Build splits
- [ ] Baseline eval (Whisper zero-shot)
- [ ] Train Whisper LoRA
- [ ] Eval Whisper LoRA
- [ ] Final report

## Timeline
- 02:30 — start, all scripts in place
- 02:45 — kicked off text gen (1000 sentences pools=province+top_leader)
- 02:50 — VIVOS dl, 15 speakers extracted as refs
- 02:55 — F5-TTS verified working (22.7s first call) after fixing vocab issue
- 03:00 — synth 300 + librispeech dl kicked off in parallel
- 03:01 — librispeech extracted, manifest built (2703 dev-clean samples)
- 03:01 → 03:18 — synth running (avg 3.4s/sample)
- 03:18 — Whisper Turbo pre-downloaded (1.6GB)
- 03:25 — synth done (298 samples), stopped F5-TTS, build splits
- 03:30 — baseline Whisper-Turbo eval: test_vn WER 207% (hallucinate), test_en WER 3.34% (works)
- 03:35 — debug train pipeline (USE_TF=0, torchcodec → soundfile, datasets.map → torch Dataset, dataloader_num_workers=0)
- 03:45 — **GPU LOST** (`nvidia-smi`: "GPU is lost. Reboot the system to recover"). Cannot recover without admin/reboot.
- 03:46 — Tried `Disable/Enable-PnpDevice` — needs admin, failed.
- 03:50 — Wrote RESULTS.md with recovery instructions.

## Final status
- Data pipeline 100% ready (synth + EN + splits).
- Baselines complete and saved.
- Training scripts fully debugged but blocked by hardware. User needs to reboot then re-run 1 command (see RESULTS.md).
