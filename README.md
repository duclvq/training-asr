# F5-TTS Vietnamese (ViVoice) — Web Demo

Web demo đơn giản cho model [`hynt/F5-TTS-Vietnamese-ViVoice`](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice).
Phục vụ qua FastAPI ở **port 8001**, bind `0.0.0.0` để truy cập được từ LAN.

## Cài đặt

```powershell
# (khuyến nghị) tạo venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# cài dependencies
pip install -r requirements.txt

# PyTorch (nếu chưa có) — chọn đúng CUDA của bạn, ví dụ CUDA 12.1:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Chạy

```powershell
python app.py
```

Truy cập:

- Máy local: <http://localhost:8001/>
- Từ máy khác trong LAN: `http://<IP-máy-này>:8001/`
  - Lấy IP: `ipconfig` → IPv4 của card đang dùng.
  - Mở firewall TCP port 8001 nếu cần:

    ```powershell
    New-NetFirewallRule -DisplayName "F5-TTS demo 8001" -Direction Inbound -Protocol TCP -LocalPort 8001 -Action Allow
    ```

Lần đầu gọi `/tts`, server sẽ tải checkpoint từ HuggingFace (vài GB) — vui lòng đợi.

## Cách dùng

1. Chọn file audio mẫu (3–10 giây, giọng càng rõ càng tốt).
2. Gõ chính xác nội dung audio đó vào ô **Nội dung audio tham chiếu**.
3. Nhập text tiếng Việt cần sinh vào ô **Text cần sinh**.
4. Bấm **Sinh audio**. Khi xong, audio player hiện ra, có nút tải `.wav`.

## Biến môi trường

- `F5_CKPT_FILENAME` — tên file checkpoint trong repo (mặc định `model_last.pt`).
- `F5_VOCAB_FILENAME` — tên file vocab (mặc định `vocab.txt`).
- `DEEPSEEK_API_KEY` — khoá DeepSeek, lưu sẵn trong `.env` (không dùng cho TTS, để dành cho phần mở rộng).

## Cấu trúc

```
training_asr/
├── app.py                  # FastAPI backend
├── static/index.html       # Frontend 1 trang
├── requirements.txt
├── .env                    # secrets (đã gitignore)
└── README.md
```
