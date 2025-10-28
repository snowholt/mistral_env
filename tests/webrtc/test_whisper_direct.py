import sys
from pathlib import Path

import pytest
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*return_token_timestamps is deprecated for WhisperFeatureExtractor.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*return_token_timestamps is deprecated for WhisperFeatureExtractor.*",
    category=UserWarning,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_SRC = ROOT_DIR / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.append(str(BACKEND_SRC))

from beautyai_inference.core.model_manager import ModelManager

AUDIO_FIXTURE = Path(__file__).resolve().parent / "laser_hair.wav"
EXPECTED_FRAGMENT = "how does laser hair removal work"


def _ensure_whisper_engine(language: str = "en"):
    print("[WHISPER-TEST] Acquiring ModelManager instance...")
    manager = ModelManager()
    print("[WHISPER-TEST] Requesting streaming whisper engine...")
    engine = manager.get_streaming_whisper(language=language)
    if engine is None:
        pytest.fail("Failed to acquire Whisper engine instance from ModelManager")

    if not getattr(engine, "model", None):
        print("[WHISPER-TEST] Loading whisper model weights...")
        loaded = engine.load_whisper_model()
        assert loaded, "Whisper engine failed to load model"
    else:
        print("[WHISPER-TEST] Whisper model already loaded; reusing instance")
    return engine


@pytest.mark.skipif(not AUDIO_FIXTURE.exists(), reason="laser_hair.wav fixture missing")
def test_whisper_large_v3_turbo_transcribes_laser_hair_clip():
    print("[WHISPER-TEST] Starting direct whisper transcription scenario...")
    engine = _ensure_whisper_engine(language="en")

    print(f"[WHISPER-TEST] Reading audio fixture from {AUDIO_FIXTURE}...")
    audio_bytes = AUDIO_FIXTURE.read_bytes()
    print("[WHISPER-TEST] Dispatching audio bytes to whisper engine...")
    transcription = engine.transcribe_audio_bytes(
        audio_bytes,
        audio_format="wav",
        language="en",
    )

    print(f"[WHISPER] Transcription: {transcription}")
    normalized = transcription.strip().lower()

    assert normalized, "Whisper transcription should not be empty"
    assert EXPECTED_FRAGMENT in normalized, (
        f"Expected fragment '{EXPECTED_FRAGMENT}' not found in transcription: '{transcription}'"
    )
