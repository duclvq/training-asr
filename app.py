"""F5-TTS Vietnamese (hynt/F5-TTS-Vietnamese-ViVoice) web demo.

Run:
    python app.py
Then open http://<LAN-IP>:5001/
"""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

HF_REPO_ID = "hynt/F5-TTS-Vietnamese-ViVoice"
CKPT_FILENAME = os.environ.get("F5_CKPT_FILENAME", "model_last.pt")
VOCAB_FILENAME = os.environ.get("F5_VOCAB_FILENAME", "vocab.txt")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

_state: dict = {}


def _download_model_files() -> tuple[str, str]:
    """Download checkpoint + vocab from HF.

    Repo `hynt/F5-TTS-Vietnamese-ViVoice` chứa `config.json` thực ra LÀ vocab
    (xem README — yêu cầu rename config.json → vocab.txt). Ta dùng trực tiếp.
    """
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model_last.pt")
    print(f"[load] checkpoint: {ckpt_path}")

    vocab_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json")
    print(f"[load] vocab (config.json renamed): {vocab_path}")

    return ckpt_path, vocab_path


def _load_model():
    """Load F5-TTS model. Done lazily on first request to keep startup fast."""
    from f5_tts.api import F5TTS  # local import so the server can boot without GPU

    ckpt_path, vocab_path = _download_model_files()

    try:
        model = F5TTS(
            model="F5TTS_Base",
            ckpt_file=ckpt_path,
            vocab_file=vocab_path,
            vocoder_name="vocos",
        )
    except TypeError:
        # Older f5-tts signature
        model = F5TTS(ckpt_file=ckpt_path, vocab_file=vocab_path)
    print("[load] F5TTS model ready.")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[boot] Will load model from {HF_REPO_ID} on first request.")
    yield
    _state.clear()


app = FastAPI(title="F5-TTS Vietnamese Demo", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model_loaded": "model" in _state}


@app.post("/tts")
async def tts(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
    gen_text: str = Form(...),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
):
    if not gen_text.strip():
        raise HTTPException(status_code=400, detail="gen_text is empty")
    if not ref_text.strip():
        raise HTTPException(status_code=400, detail="ref_text is empty")

    # Lazy-load on first request.
    if "model" not in _state:
        try:
            _state["model"] = _load_model()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Model load failed: {e}") from e
    model = _state["model"]

    # Persist uploaded audio to a temp file (F5-TTS API takes a path).
    suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await ref_audio.read())
        ref_path = tmp_in.name

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    try:
        wav, sr, _ = model.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=out_path,
            remove_silence=remove_silence,
            speed=speed,
        )
    except TypeError:
        # Older f5-tts signatures may not accept `speed` etc. — retry minimal.
        wav, sr, _ = model.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=out_path,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    finally:
        try:
            os.unlink(ref_path)
        except OSError:
            pass

    # Read output back as bytes and stream to client.
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    try:
        os.unlink(out_path)
    except OSError:
        pass

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="output.wav"'},
    )


# Static last so it does not override the routes above.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
