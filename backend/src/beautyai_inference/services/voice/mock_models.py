"""
Mock Models for VAD Testing

This module provides mock implementations of Whisper, LLM, and TTS models
for testing WebRTC + VAD functionality without loading heavy models.

Author: BeautyAI Framework
Date: October 29, 2025
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MockTranscriptionResult:
    """Mock transcription result from Whisper."""
    text: str
    language: str
    confidence: float
    duration: float
    segments: List[Dict[str, Any]]


class MockWhisperModel:
    """
    Mock Whisper model for VAD testing.
    
    Returns dummy transcriptions without loading actual model.
    Useful for testing audio pipeline and VAD behavior.
    """
    
    def __init__(
        self,
        model_id: str = "mock-whisper",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """Initialize mock Whisper model."""
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.is_loaded = False
        self.transcription_count = 0
        
        logger.info(
            f"[MOCK] Whisper model initialized: {model_id} "
            f"(device={device}, compute={compute_type})"
        )
    
    async def load(self) -> bool:
        """Mock model loading."""
        logger.info(f"[MOCK] Loading Whisper model: {self.model_id}")
        await self._simulate_loading(0.1)  # Simulate 100ms load time
        self.is_loaded = True
        logger.info(f"[MOCK] Whisper model loaded successfully")
        return True
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        **kwargs
    ) -> MockTranscriptionResult:
        """
        Mock transcription of audio data.
        
        Returns realistic-looking transcription based on audio duration.
        """
        if not self.is_loaded:
            raise RuntimeError("Mock Whisper model not loaded")
        
        self.transcription_count += 1
        
        # Calculate audio duration (assuming 16kHz mono 16-bit PCM)
        audio_duration = len(audio_data) / (16000 * 2)
        
        # Generate mock transcription based on language
        if language == "ar":
            mock_text = f"[MOCK AR] نص محاكاة رقم {self.transcription_count}"
        else:
            mock_text = f"[MOCK EN] Mock transcription #{self.transcription_count}"
        
        # Simulate processing time (10% of audio duration)
        await self._simulate_loading(audio_duration * 0.1)
        
        result = MockTranscriptionResult(
            text=mock_text,
            language=language or "en",
            confidence=0.95,
            duration=audio_duration,
            segments=[{
                "id": 0,
                "start": 0.0,
                "end": audio_duration,
                "text": mock_text,
                "confidence": 0.95
            }]
        )
        
        logger.info(
            f"[MOCK] Whisper transcription #{self.transcription_count}: "
            f"'{mock_text}' (duration={audio_duration:.2f}s, language={language})"
        )
        
        return result
    
    async def unload(self):
        """Mock model unloading."""
        logger.info(f"[MOCK] Unloading Whisper model: {self.model_id}")
        self.is_loaded = False
    
    @staticmethod
    async def _simulate_loading(duration: float):
        """Simulate async processing delay."""
        import asyncio
        await asyncio.sleep(duration)


class MockLLMModel:
    """
    Mock LLM model for VAD testing.
    
    Returns dummy responses without loading actual model.
    """
    
    def __init__(
        self,
        model_id: str = "mock-llm",
        device: str = "cpu"
    ):
        """Initialize mock LLM model."""
        self.model_id = model_id
        self.device = device
        self.is_loaded = False
        self.response_count = 0
        
        logger.info(f"[MOCK] LLM model initialized: {model_id} (device={device})")
    
    async def load(self) -> bool:
        """Mock model loading."""
        logger.info(f"[MOCK] Loading LLM model: {self.model_id}")
        await MockWhisperModel._simulate_loading(0.2)
        self.is_loaded = True
        logger.info(f"[MOCK] LLM model loaded successfully")
        return True
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Mock text generation."""
        if not self.is_loaded:
            raise RuntimeError("Mock LLM model not loaded")
        
        self.response_count += 1
        
        # Simulate processing time
        await MockWhisperModel._simulate_loading(0.3)
        
        # Generate mock response
        mock_response = (
            f"[MOCK LLM RESPONSE #{self.response_count}] "
            f"This is a mock response to: '{prompt[:50]}...'"
        )
        
        logger.info(
            f"[MOCK] LLM response #{self.response_count}: "
            f"'{mock_response}' (prompt_len={len(prompt)} chars)"
        )
        
        return mock_response
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs
    ) -> AsyncIterator[str]:
        """Mock streaming text generation."""
        if not self.is_loaded:
            raise RuntimeError("Mock LLM model not loaded")
        
        self.response_count += 1
        response = (
            f"[MOCK LLM STREAM #{self.response_count}] "
            f"Mock streaming response to prompt."
        )
        
        # Yield words one at a time with delay
        words = response.split()
        for word in words:
            await MockWhisperModel._simulate_loading(0.05)
            yield word + " "
    
    async def unload(self):
        """Mock model unloading."""
        logger.info(f"[MOCK] Unloading LLM model: {self.model_id}")
        self.is_loaded = False


class MockTTSModel:
    """
    Mock TTS model for VAD testing.
    
    Returns dummy audio without loading actual model.
    """
    
    def __init__(
        self,
        model_id: str = "mock-tts",
        voice: str = "mock-voice"
    ):
        """Initialize mock TTS model."""
        self.model_id = model_id
        self.voice = voice
        self.is_loaded = False
        self.synthesis_count = 0
        
        logger.info(
            f"[MOCK] TTS model initialized: {model_id} (voice={voice})"
        )
    
    async def load(self) -> bool:
        """Mock model loading."""
        logger.info(f"[MOCK] Loading TTS model: {self.model_id}")
        await MockWhisperModel._simulate_loading(0.1)
        self.is_loaded = True
        logger.info(f"[MOCK] TTS model loaded successfully")
        return True
    
    async def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Mock text-to-speech synthesis.
        
        Returns dummy PCM audio data.
        """
        if not self.is_loaded:
            raise RuntimeError("Mock TTS model not loaded")
        
        self.synthesis_count += 1
        
        # Simulate processing time (50ms per 10 characters)
        text_duration = len(text) / 10 * 0.05
        await MockWhisperModel._simulate_loading(text_duration)
        
        # Generate dummy audio (1 second of silence at 16kHz mono 16-bit)
        audio_duration_sec = 1.0
        sample_rate = 16000
        num_samples = int(audio_duration_sec * sample_rate)
        
        # Generate very quiet sine wave (not complete silence)
        t = np.linspace(0, audio_duration_sec, num_samples)
        audio_float = np.sin(2 * np.pi * 440 * t) * 0.01  # 440Hz at 1% volume
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        logger.info(
            f"[MOCK] TTS synthesis #{self.synthesis_count}: "
            f"'{text[:50]}...' -> {len(audio_bytes)} bytes "
            f"(~{audio_duration_sec:.1f}s audio)"
        )
        
        return audio_bytes
    
    async def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """Mock streaming TTS synthesis."""
        if not self.is_loaded:
            raise RuntimeError("Mock TTS model not loaded")
        
        self.synthesis_count += 1
        
        # Generate audio in chunks
        chunk_size = 3200  # 0.1s of audio at 16kHz
        total_audio = await self.synthesize(text, language, **kwargs)
        
        # Yield audio in chunks
        for i in range(0, len(total_audio), chunk_size):
            chunk = total_audio[i:i + chunk_size]
            await MockWhisperModel._simulate_loading(0.05)
            yield chunk
    
    async def unload(self):
        """Mock model unloading."""
        logger.info(f"[MOCK] Unloading TTS model: {self.model_id}")
        self.is_loaded = False


class MockModelFactory:
    """
    Factory for creating mock models.
    
    Used when VAD_TEST_MODE=1 environment variable is set.
    """
    
    @staticmethod
    def create_whisper_model(**kwargs) -> MockWhisperModel:
        """Create mock Whisper model."""
        return MockWhisperModel(**kwargs)
    
    @staticmethod
    def create_llm_model(**kwargs) -> MockLLMModel:
        """Create mock LLM model."""
        return MockLLMModel(**kwargs)
    
    @staticmethod
    def create_tts_model(**kwargs) -> MockTTSModel:
        """Create mock TTS model."""
        return MockTTSModel(**kwargs)


# Convenience function for checking if mock mode is enabled
def is_vad_test_mode() -> bool:
    """Check if VAD test mode is enabled via environment variable."""
    import os
    return os.getenv("VAD_TEST_MODE", "0") in {"1", "true", "True", "TRUE"}


def get_model_factory():
    """
    Get appropriate model factory based on test mode.
    
    Returns MockModelFactory if VAD_TEST_MODE=1, otherwise None.
    """
    if is_vad_test_mode():
        logger.info("[MOCK] VAD test mode enabled - using mock models")
        return MockModelFactory()
    return None
