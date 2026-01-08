import numpy as np
import soundfile as sf
from pathlib import Path

def log_pcm_stats(label: str, audio: np.ndarray, sr: int, logger=None, dump_path: str | Path | None = None):
    """Log and optionally write diagnostic WAV with duration, peak, RMS, dtype, channels."""
    # Normalize shapes
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        ch = audio.shape[0]
    elif audio.ndim == 2:
        ch = audio.shape[1]
    else:
        ch = 1

    samples = len(audio.flatten())
    duration_s = samples / float(sr)
    peak = float(np.abs(audio).max()) if samples else 0.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2))) if samples else 0.0
    msg = (f"[PCM] label={label} sr={sr} ch={ch} dtype={audio.dtype} "
           f"samples={samples} duration_s={duration_s:6.3f} "
           f"peak={peak:7.4f} rms={rms:7.4f}")
    (logger.info if logger else print)(msg)

    # Optional dump
    if dump_path:
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            sf.write(dump_path, audio, sr)
            (logger.info if logger else print)(f"[PCM] wrote {dump_path}")
        except Exception as e:
            (logger.warning if logger else print)(f"[PCM] failed to write {dump_path}: {e}")
