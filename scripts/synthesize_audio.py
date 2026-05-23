"""Synthesize audio cho mọi câu trong generated_sentences.jsonl bằng F5-TTS API.

Cần server F5-TTS đang chạy (mặc định http://localhost:8001).

Audio đầu ra:
    data/audio/synth/<id>.wav

Manifest (tương thích HF datasets):
    data/manifests/synth_manifest.jsonl
    {"audio": "...wav", "text": "...", "duration": <s>, "source": "vn_augmented",
     "vietnamese_terms": [...], "category": "...", "style": "...", "context": "..."}

Cần file reference audio + reference text:
    --ref-audio path/to/ref.wav
    --ref-text "Nội dung audio mẫu đúng nguyên văn"

Lưu ý: F5-TTS-Vietnamese-ViVoice là model TTS tiếng Việt, mà câu của ta là tiếng Anh chứa
tên VN. Khi đọc câu tiếng Anh model có thể đọc theo lối tiếng Việt (phát âm gần với cách
một người Việt đọc tiếng Anh) — đó là intent: phục vụ ASR học giọng Việt đọc tiếng Anh.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import soundfile as sf
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "generated_sentences.jsonl"
DEFAULT_OUT_DIR = DATA_DIR / "audio" / "synth"
DEFAULT_MANIFEST = DATA_DIR / "manifests" / "synth_manifest.jsonl"


def load_existing_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    ids = set()
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(Path(json.loads(line)["audio"]).stem)
            except Exception:
                pass
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL câu cần đọc")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="thư mục lưu wav")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest JSONL")
    ap.add_argument("--api", default="http://localhost:8001/tts", help="endpoint F5-TTS")
    ap.add_argument("--ref-audio", type=Path, default=None, help="single ref audio")
    ap.add_argument("--ref-text", default=None, help="transcript of ref-audio")
    ap.add_argument("--refs-index", type=Path, default=None,
                    help="data/refs/index.json — cycle nhiều ref cho diversity")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--remove-silence", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = không giới hạn")
    ap.add_argument("--retries", type=int, default=2, help="số lần retry mỗi request")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    # Build the list of (ref_audio_path, ref_text) — either from --refs-index or single --ref-audio.
    refs: list[tuple[Path, str]] = []
    if args.refs_index:
        idx = json.loads(args.refs_index.read_text(encoding="utf-8"))
        for r in idx:
            wav_p = Path(r["wav"])
            text_p = Path(r["txt"])
            if wav_p.exists() and text_p.exists():
                refs.append((wav_p, text_p.read_text(encoding="utf-8").strip()))
        if not refs:
            print("ERROR: refs-index loaded but no usable refs.", file=sys.stderr)
            sys.exit(1)
        print(f"[refs] cycling through {len(refs)} voices")
    elif args.ref_audio and args.ref_text:
        if not args.ref_audio.exists():
            print(f"ERROR: ref-audio {args.ref_audio} not found.", file=sys.stderr)
            sys.exit(1)
        refs.append((args.ref_audio, args.ref_text))
    else:
        print("ERROR: pass either --refs-index OR (--ref-audio + --ref-text)", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_existing_ids(args.manifest)
    print(f"[resume] {len(done_ids)} samples already in manifest, skipping those")

    items = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["id"] in done_ids:
                continue
            items.append(r)
            if args.limit and len(items) >= args.limit:
                break
    print(f"[plan] {len(items)} sentences to synthesize")

    out_manifest = args.manifest.open("a", encoding="utf-8")

    pbar = tqdm(items, unit="sample")
    n_ok = n_fail = 0
    # Pre-load ref bytes so we don't re-read from disk.
    ref_cache = {p: p.read_bytes() for p, _ in refs}
    for i, r in enumerate(pbar):
        sample_id = r["id"]
        out_wav = args.out_dir / f"{sample_id}.wav"
        ref_audio_path, ref_text = refs[i % len(refs)]
        ref_bytes = ref_cache[ref_audio_path]

        wav_bytes = None
        for attempt in range(1, args.retries + 2):
            files = {"ref_audio": (ref_audio_path.name, ref_bytes, "audio/wav")}
            data = {
                "ref_text": ref_text,
                "gen_text": r["sentence"],
                "speed": str(args.speed),
                "remove_silence": "true" if args.remove_silence else "false",
            }
            try:
                resp = requests.post(args.api, files=files, data=data, timeout=600)
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("audio"):
                    wav_bytes = resp.content
                    break
                err = resp.text[:200]
                pbar.write(f"[{sample_id}] attempt {attempt}: HTTP {resp.status_code} {err}")
            except Exception as e:  # noqa: BLE001
                pbar.write(f"[{sample_id}] attempt {attempt}: {e}")
            time.sleep(2 * attempt)

        if wav_bytes is None:
            n_fail += 1
            continue

        out_wav.write_bytes(wav_bytes)

        try:
            info = sf.info(str(out_wav))
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            duration = -1.0

        rec = {
            "audio": str(out_wav.resolve()),
            "text": r["sentence"],
            "duration": round(duration, 3),
            "source": "vn_augmented",
            "vietnamese_terms": r.get("vietnamese_terms", []),
            "category": r.get("category"),
            "style": r.get("seed", {}).get("style"),
            "context": r.get("seed", {}).get("context"),
            "ref_voice": ref_audio_path.stem,
        }
        out_manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_manifest.flush()
        n_ok += 1
        pbar.set_postfix(ok=n_ok, fail=n_fail)

    out_manifest.close()
    print(f"[done] ok={n_ok} fail={n_fail} -> manifest {args.manifest}")


if __name__ == "__main__":
    main()
