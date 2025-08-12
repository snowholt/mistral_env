"""
Batch transcribe saved WebM/WAV turn segments for debugging Arabic/English recognition.

Usage:
  source backend/venv/bin/activate
  python vad_testting_temp/transcribe_debug_segments.py --model tiny --lang ar

Options:
  --dir   Directory containing turn_segment_*.webm/.wav (default vad_testting_temp)
  --model Whisper model size/name (tiny, base, small, medium, large, or path)
  --lang  Optional language hint (ar, en, etc.)

Outputs transcription_report.json in the same directory.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import subprocess
import json


def find_segments(debug_dir: Path):
    return sorted(list(debug_dir.glob("turn_segment_*.webm")) + list(debug_dir.glob("turn_segment_*.wav")))


def ensure_wav(src_path: Path) -> Path:
    if src_path.suffix.lower() == ".wav":
        return src_path
    wav_path = src_path.with_suffix(".wav")
    if not wav_path.exists():
        cmd = [
            "ffmpeg", "-y", "-i", str(src_path),
            "-ar", "16000", "-ac", "1", str(wav_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path if wav_path.exists() else src_path


def transcribe_whisper(wav_path: Path, model: str, lang: str):
    try:
        import whisper  # type: ignore
    except ImportError:
        print("[WARN] whisper not installed. Install: pip install -U openai-whisper")
        return ""
    try:
        m = whisper.load_model(model)
        result = m.transcribe(str(wav_path), language=lang if lang else None)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[ERR] Transcription failed for {wav_path.name}: {e}")
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="vad_testting_temp", help="Debug audio directory")
    ap.add_argument("--model", default="small", help="Whisper model size/name")
    ap.add_argument("--lang", default="", help="Optional language hint (ar/en)")
    args = ap.parse_args()

    debug_dir = Path(args.dir).resolve()
    if not debug_dir.exists():
        print(f"Directory not found: {debug_dir}")
        sys.exit(1)

    segments = find_segments(debug_dir)
    if not segments:
        print("No turn_segment_* files found.")
        return

    print(f"Found {len(segments)} segment(s). Transcribing...")
    report = []
    for seg in segments:
        wav = ensure_wav(seg)
        text = transcribe_whisper(wav, args.model, args.lang)
        entry = {
            "file": seg.name,
            "wav_file": wav.name,
            "size_bytes": seg.stat().st_size,
            "transcription": text,
        }
        report.append(entry)
        print(f"{seg.name} -> {text}")

    out_path = debug_dir / "transcription_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
