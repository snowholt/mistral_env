"""Unit tests for WebRTCAudioProcessor edge cases."""

import numpy as np
import pytest

from beautyai_inference.services.voice.webrtc_audio_processor import (
    AudioProcessingConfig,
    WebRTCAudioProcessor,
)


class _DummyLayout:
    """Simple container mimicking aiortc layout."""

    def __init__(self, channels: int) -> None:
        self.channels = [None] * max(channels, 0)


class _DummyFrame:
    """Minimal AudioFrame replacement for testing."""

    def __init__(self, audio: np.ndarray, sample_rate: int, channels: int | None) -> None:
        self._audio = audio.astype(np.float32)
        self.sample_rate = sample_rate
        self.layout = _DummyLayout(channels) if channels else None

    def to_ndarray(self) -> np.ndarray:  # pragma: no cover - trivial passthrough
        return self._audio


@pytest.mark.asyncio
async def test_process_audio_frame_handles_zero_sample_rate() -> None:
    """Frames lacking sample-rate metadata should fall back safely."""

    captured: list[tuple[bytes, dict]] = []

    async def on_chunk(chunk: bytes, metadata: dict) -> None:
        captured.append((chunk, metadata))

    config = AudioProcessingConfig(enable_level_monitoring=False)
    processor = WebRTCAudioProcessor(
        peer_id="peer-zero",
        config=config,
        on_audio_chunk=on_chunk,
    )

    stereo_audio = np.array([[0.0, 0.2, -0.3], [0.1, -0.1, 0.4]], dtype=np.float32)
    frame = _DummyFrame(audio=stereo_audio, sample_rate=0, channels=2)

    await processor._process_audio_frame(frame)

    assert captured, "audio chunk should be delivered even when metadata is invalid"
    chunk_bytes, metadata = captured[0]

    assert len(chunk_bytes) > 0, "PCM payload should be produced"
    assert metadata["sample_rate"] == config.target_sample_rate
    assert processor.metrics.sample_rate == config.target_sample_rate
    assert processor.metrics.frames_processed == 1


@pytest.mark.asyncio
async def test_process_audio_frame_handles_missing_layout() -> None:
    """Frames without layout information should default to config channels."""

    captured: list[tuple[bytes, dict]] = []

    async def on_chunk(chunk: bytes, metadata: dict) -> None:
        captured.append((chunk, metadata))

    config = AudioProcessingConfig(enable_level_monitoring=False)
    processor = WebRTCAudioProcessor(
        peer_id="peer-layout",
        config=config,
        on_audio_chunk=on_chunk,
    )

    mono_audio = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
    frame = _DummyFrame(audio=mono_audio, sample_rate=config.target_sample_rate, channels=None)

    await processor._process_audio_frame(frame)

    assert captured, "audio chunk should be delivered"
    _, metadata = captured[0]

    assert metadata["channels"] == config.target_channels
    assert processor.metrics.channels == config.target_channels
    assert processor.metrics.frames_processed == 1
