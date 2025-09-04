"""
Edge TTS Text-to-Speech engine for BeautyAI Framework.
Uses Microsoft Edge TTS for high-quality multilingual TTS generation.
This is compatible with Python 3.12+ .
"""

import logging
import os
import asyncio
import concurrent.futures
import threading
import io
import wave
from typing import Dict, Any, Optional, List, AsyncGenerator, Union
from pathlib import Path

from ..core.model_interface import ModelInterface
from ..config.config_manager import ModelConfig

logger = logging.getLogger(__name__)

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    logger.warning("edge-tts library not available. Install with: pip install edge-tts")
    EDGE_TTS_AVAILABLE = False


class EdgeTTSEngine(ModelInterface):
    """
    Edge TTS Text-to-Speech engine using Microsoft Edge TTS.
    
    Features:
    - Multilingual support (Arabic, English, and many more)
    - High-quality neural synthesis
    - Python 3.12+ compatible
    - No GPU required
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the Edge TTS engine with a model configuration."""
        if not EDGE_TTS_AVAILABLE:
            raise ImportError(
                "edge-tts library is required. Install with: pip install edge-tts"
            )
        
        self.config = model_config
        self.model = None
        
        # Initialize voice mappings first
        self._init_voice_mappings()
        
        # Performance optimizations
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)  # Increased for better parallelism
        self._voice_cache = {}  # Cache voice objects for reuse
        self._loop_cache = {}   # Cache event loops per thread
        
        # Pre-warm the most common voices for faster access
        self._preload_common_voices()
        
        # Streaming support for duplex voice
        self._chunk_size_ms = 40  # 40ms chunks for low latency
        self._streaming_sample_rate = 16000  # Target sample rate for streaming

    def _init_voice_mappings(self):
        """Initialize voice mappings."""
        # Language to voice mapping for Edge TTS
        self.voice_mapping = {
            "en": "en-US-AriaNeural",      # English (US) - Female
            "ar": "ar-SA-ZariyahNeural",   # Arabic (Saudi Arabia) - Female
            "es": "es-ES-ElviraNeural",    # Spanish (Spain) - Female
            "fr": "fr-FR-DeniseNeural",    # French (France) - Female
            "de": "de-DE-KatjaNeural",     # German (Germany) - Female
            "it": "it-IT-ElsaNeural",      # Italian (Italy) - Female
            "pt": "pt-BR-FranciscaNeural", # Portuguese (Brazil) - Female
            "pl": "pl-PL-ZofiaNeural",     # Polish (Poland) - Female
            "tr": "tr-TR-EmelNeural",      # Turkish (Turkey) - Female
            "ru": "ru-RU-SvetlanaNeural",  # Russian (Russia) - Female
            "nl": "nl-NL-ColetteNeural",   # Dutch (Netherlands) - Female
            "cs": "cs-CZ-VlastaNeural",    # Czech (Czech Republic) - Female
            "zh": "zh-CN-XiaoxiaoNeural",  # Chinese (China) - Female
            "ja": "ja-JP-NanamiNeural",    # Japanese (Japan) - Female
            "hu": "hu-HU-NoemiNeural",     # Hungarian (Hungary) - Female
            "ko": "ko-KR-SunHiNeural",     # Korean (Korea) - Female
        }
        
        # Alternative male voices
        self.male_voice_mapping = {
            "en": "en-US-ChristopherNeural",
            "ar": "ar-SA-HamedNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-ConradNeural",
            "it": "it-IT-DiegoNeural",
            "pt": "pt-BR-AntonioNeural",
            "pl": "pl-PL-MarekNeural",
            "tr": "tr-TR-AhmetNeural",
            "ru": "ru-RU-DmitryNeural",
            "nl": "nl-NL-MaartenNeural",
            "cs": "cs-CZ-AntoninNeural",
            "zh": "zh-CN-YunxiNeural",
            "ja": "ja-JP-KeitaNeural",
            "hu": "hu-HU-TamasNeural",
            "ko": "ko-KR-InJoonNeural",
        }

    def _preload_common_voices(self):
        """Pre-cache the most commonly used voices."""
        common_voices = [
            ("en", "female", None),
            ("ar", "female", None),
            ("en", "male", None),
            ("ar", "male", None)
        ]
        for lang, gender, speaker in common_voices:
            self._get_voice(lang, speaker, gender)

    def load_model(self) -> bool:
        """Load the Edge TTS model (no actual loading required)."""
        logger.info("Edge TTS model ready - no loading required")
        logger.info("Available languages: " + ", ".join(self.voice_mapping.keys()))
        return True

    def unload_model(self) -> bool:
        """Unload the model and cleanup resources."""
        logger.info("Edge TTS model unloaded - cleaning up resources")
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        self._voice_cache.clear()
        self._loop_cache.clear()
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate speech from text."""
        return self.text_to_speech(prompt, **kwargs)

    def text_to_speech(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        output_path: Optional[str] = None,
        gender: str = "female",
        **kwargs
    ) -> str:
        """
        Convert text to speech using Edge TTS with performance optimizations.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific voice to use (overrides language mapping)
            output_path: Path to save the audio file (optional)
            gender: Voice gender ("female" or "male")
            **kwargs: Additional parameters (ignored for Edge TTS compatibility)
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Fast text preprocessing for better performance
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            # Optimize for short texts (< 200 chars) - use sync generation
            if len(text) < 200:
                return self._generate_fast_sync(text, language, speaker_voice, output_path, gender)
            # For longer texts, use parallel processing
            else:
                return self._generate_parallel(text, language, speaker_voice, output_path, gender)
                
        except Exception as e:
            logger.error(f"Error during Edge TTS generation: {e}")
            raise

    def _generate_fast_sync(self, text: str, language: str, speaker_voice: Optional[str], 
                           output_path: Optional[str], gender: str) -> str:
        """Fast synchronous generation for short texts."""
        # Select voice (cached)
        voice = self._get_voice(language, speaker_voice, gender)
        
        # Create output path
        if output_path is None:
            output_path = self._create_output_path(text, language)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.debug(f"Fast TTS generation: {len(text)} chars, voice: {voice}")
        
        # Use optimized async execution with shorter timeout
        future = self._executor.submit(self._run_async_in_thread_fast, text, voice, output_path)
        future.result(timeout=8)  # Reduced timeout for faster failure detection
        
        logger.debug(f"Edge TTS audio saved to: {output_path}")
        return output_path

    def _run_async_in_thread_fast(self, text: str, voice: str, output_path: str) -> None:
        """Optimized async TTS generation for short texts."""
        # Use a new event loop for maximum performance
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async generation with optimization
            loop.run_until_complete(self._generate_tts_async_fast(text, voice, output_path))
        finally:
            loop.close()

    async def _generate_tts_async_fast(self, text: str, voice: str, output_path: str):
        """Ultra-fast async TTS generation with minimal overhead."""
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def _generate_parallel(self, text: str, language: str, speaker_voice: Optional[str], 
                          output_path: Optional[str], gender: str) -> str:
        """Parallel generation for longer texts by splitting into chunks."""
        # Split text into sentences for parallel processing
        sentences = self._split_text_smart(text)
        
        if len(sentences) == 1:
            # Single sentence, use fast sync
            return self._generate_fast_sync(text, language, speaker_voice, output_path, gender)
        
        voice = self._get_voice(language, speaker_voice, gender)
        
        if output_path is None:
            output_path = self._create_output_path(text, language)
        
        logger.debug(f"Parallel TTS generation: {len(sentences)} chunks, voice: {voice}")
        
        # Generate all chunks in parallel
        chunk_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(sentences))) as executor:
            futures = []
            for i, sentence in enumerate(sentences):
                chunk_path = output_path.replace('.wav', f'_chunk_{i}.wav')
                future = executor.submit(self._run_async_in_thread, sentence, voice, chunk_path)
                futures.append((future, chunk_path))
            
            # Wait for all chunks
            for future, chunk_path in futures:
                future.result(timeout=15)
                chunk_paths.append(chunk_path)
        
        # Combine audio chunks
        self._combine_audio_files(chunk_paths, output_path)
        
        # Cleanup chunk files
        for chunk_path in chunk_paths:
            try:
                os.remove(chunk_path)
            except:
                pass
        
        logger.info(f"Edge TTS audio saved to: {output_path}")
        return output_path

    def _get_voice(self, language: str, speaker_voice: Optional[str], gender: str) -> str:
        """Get voice with caching for performance."""
        cache_key = f"{language}_{gender}_{speaker_voice}"
        
        if cache_key in self._voice_cache:
            return self._voice_cache[cache_key]
        
        if speaker_voice is None:
            if gender.lower() == "male" and language in self.male_voice_mapping:
                voice = self.male_voice_mapping[language]
            else:
                voice = self.voice_mapping.get(language, "en-US-AriaNeural")
        else:
            voice = speaker_voice
        
        self._voice_cache[cache_key] = voice
        return voice

    def _create_output_path(self, text: str, language: str) -> str:
        """Create optimized output path."""
        tests_dir = Path(__file__).parent.parent.parent / "voice_tests"
        tests_dir.mkdir(exist_ok=True)
        
        # Use faster hash for file naming
        text_hash = abs(hash(text)) % 100000
        return str(tests_dir / f"edge_tts_{language}_{text_hash}.wav")

    def _split_text_smart(self, text: str) -> List[str]:
        """Smart text splitting for parallel processing."""
        # Split on sentence boundaries but keep reasonable chunk sizes
        import re
        
        # Split on periods, exclamation marks, question marks
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have very short sentences, combine them
        if len(sentences) > 1 and all(len(s) < 50 for s in sentences):
            # Combine short sentences into ~100 char chunks
            combined = []
            current = ""
            for sentence in sentences:
                if len(current + sentence) < 100:
                    current += sentence + ". "
                else:
                    if current:
                        combined.append(current.strip())
                    current = sentence + ". "
            if current:
                combined.append(current.strip())
            return combined
        
        return sentences

    def _run_async_in_thread(self, text: str, voice: str, output_path: str) -> None:
        """Run async TTS generation in a thread with event loop."""
        thread_id = threading.get_ident()
        
        # Reuse event loop per thread for better performance
        if thread_id not in self._loop_cache:
            self._loop_cache[thread_id] = asyncio.new_event_loop()
        
        loop = self._loop_cache[thread_id]
        asyncio.set_event_loop(loop)
        
        # Run the async generation
        loop.run_until_complete(self._generate_tts_async_optimized(text, voice, output_path))

    async def _generate_tts_async_optimized(self, text: str, voice: str, output_path: str):
        """Optimized async TTS generation."""
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def _combine_audio_files(self, chunk_paths: List[str], output_path: str):
        """Combine multiple audio files into one."""
        try:
            import wave
            
            # Get parameters from first file
            with wave.open(chunk_paths[0], 'rb') as first_wave:
                params = first_wave.getparams()
            
            # Combine all chunks
            with wave.open(output_path, 'wb') as output_wave:
                output_wave.setparams(params)
                
                for chunk_path in chunk_paths:
                    with wave.open(chunk_path, 'rb') as chunk_wave:
                        output_wave.writeframes(chunk_wave.readframes(chunk_wave.getnframes()))
                        
        except Exception as e:
            logger.warning(f"Failed to combine audio files: {e}")
            # Fallback: just use the first chunk
            import shutil
            shutil.copy2(chunk_paths[0], output_path)

    async def stream_tts_chunks(
        self, 
        text: str, 
        language: str = "en",
        speaker_voice: Optional[str] = None,
        gender: str = "female",
        chunk_size_ms: int = 40,
        target_sample_rate: int = 16000,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio as small chunks for duplex voice communication.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, ar, es, etc.)
            speaker_voice: Specific voice to use (overrides language mapping)
            gender: Voice gender ("female" or "male")
            chunk_size_ms: Size of each audio chunk in milliseconds (default: 40ms)
            target_sample_rate: Target sample rate for output (default: 16000)
            **kwargs: Additional parameters
            
        Yields:
            bytes: PCM16 audio chunks ready for WebSocket streaming
        """
        try:
            # Fast text preprocessing
            text = text.strip()
            if not text:
                return
            
            # Select voice
            voice = self._get_voice(language, speaker_voice, gender)
            
            logger.debug(f"Streaming TTS: {len(text)} chars, voice: {voice}, chunks: {chunk_size_ms}ms")
            
            # Create communicate object
            communicate = edge_tts.Communicate(text, voice)
            
            # Stream audio data
            audio_buffer = io.BytesIO()
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
                    
                    # Process accumulated audio into chunks
                    audio_data = audio_buffer.getvalue()
                    if len(audio_data) >= self._calculate_min_buffer_size(chunk_size_ms, target_sample_rate):
                        # Convert to PCM16 and yield chunks
                        async for pcm_chunk in self._process_audio_to_chunks(
                            audio_data, chunk_size_ms, target_sample_rate
                        ):
                            yield pcm_chunk
                        
                        # Reset buffer
                        audio_buffer = io.BytesIO()
            
            # Process any remaining audio
            remaining_audio = audio_buffer.getvalue()
            if remaining_audio:
                async for pcm_chunk in self._process_audio_to_chunks(
                    remaining_audio, chunk_size_ms, target_sample_rate
                ):
                    yield pcm_chunk
                    
        except Exception as e:
            logger.error(f"Error during streaming TTS: {e}")
            raise

    def _calculate_min_buffer_size(self, chunk_size_ms: int, sample_rate: int) -> int:
        """Calculate minimum buffer size needed before processing chunks."""
        # Edge TTS typically outputs at 24kHz, so we need enough data for processing
        return int((24000 * chunk_size_ms * 2) / 1000)  # 2 bytes per sample

    async def _process_audio_to_chunks(
        self, 
        audio_data: bytes, 
        chunk_size_ms: int, 
        target_sample_rate: int
    ) -> AsyncGenerator[bytes, None]:
        """Process raw audio data into PCM16 chunks at target sample rate."""
        try:
            # Convert to WAV format first
            wav_buffer = io.BytesIO()
            
            # Edge TTS outputs 24kHz 16-bit mono by default
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # Edge TTS default
                wav_file.writeframes(audio_data)
            
            wav_buffer.seek(0)
            
            # Read back as wave
            with wave.open(wav_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                
                # Convert to target sample rate if needed
                if wav_file.getframerate() != target_sample_rate:
                    frames = await self._resample_audio(
                        frames, wav_file.getframerate(), target_sample_rate
                    )
                
                # Split into chunks
                samples_per_chunk = int((target_sample_rate * chunk_size_ms) / 1000)
                bytes_per_chunk = samples_per_chunk * 2  # 16-bit = 2 bytes per sample
                
                for i in range(0, len(frames), bytes_per_chunk):
                    chunk = frames[i:i + bytes_per_chunk]
                    if len(chunk) >= bytes_per_chunk // 2:  # At least half a chunk
                        yield chunk
                        
        except Exception as e:
            logger.error(f"Error processing audio chunks: {e}")
            # Yield silence as fallback
            samples_per_chunk = int((target_sample_rate * chunk_size_ms) / 1000)
            silence = b'\x00' * (samples_per_chunk * 2)
            yield silence

    async def _resample_audio(self, audio_data: bytes, source_rate: int, target_rate: int) -> bytes:
        """Simple audio resampling."""
        if source_rate == target_rate:
            return audio_data
        
        # Simple linear interpolation resampling
        import array
        source_samples = array.array('h')  # signed short
        source_samples.frombytes(audio_data)
        
        # Calculate ratio
        ratio = source_rate / target_rate
        target_length = int(len(source_samples) / ratio)
        
        target_samples = array.array('h')
        for i in range(target_length):
            source_idx = i * ratio
            base_idx = int(source_idx)
            
            if base_idx + 1 < len(source_samples):
                # Linear interpolation
                frac = source_idx - base_idx
                sample = source_samples[base_idx] * (1 - frac) + source_samples[base_idx + 1] * frac
                target_samples.append(int(sample))
            elif base_idx < len(source_samples):
                target_samples.append(source_samples[base_idx])
        
        return target_samples.tobytes()

    def get_available_speakers(self, language: str = None) -> List[str]:
        """Get available speakers for the specified language."""
        if language and language in self.voice_mapping:
            return [self.voice_mapping[language], self.male_voice_mapping.get(language, "")]
        return list(self.voice_mapping.values())

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.voice_mapping.keys())

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Not applicable for TTS engine."""
        raise NotImplementedError("Chat not supported for TTS engine")

    def benchmark(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Benchmark TTS generation."""
        import time
        start_time = time.time()
        
        result_path = self.text_to_speech(prompt, **kwargs)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate characters per second
        chars_per_second = len(prompt) / generation_time if generation_time > 0 else 0
        
        return {
            "generation_time": generation_time,
            "characters_per_second": chars_per_second,
            "audio_file": result_path,
            "engine": "edge-tts"
        }

    def chat_stream(self, messages: List[Dict[str, str]], callback=None, **kwargs) -> str:
        """Not applicable for TTS engine."""
        raise NotImplementedError("Chat streaming not supported for TTS engine")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            "memory_used_mb": 0.0,  # Edge TTS doesn't use significant memory
            "gpu_memory_used_mb": 0.0,  # Edge TTS doesn't use GPU
        }

    def is_model_loaded(self) -> bool:
        """Check if model is loaded (always True for Edge TTS)."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "Microsoft Edge TTS",
            "type": "neural_tts",
            "languages": self.get_supported_languages(),
            "voices": len(self.voice_mapping),
            "gpu_required": False,
            "python_compatibility": "3.12+"
        }
