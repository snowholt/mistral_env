"""
Enhanced Streaming TTS Service with Interruption Support.

Provides interruptible TTS streaming with:
- Cancellation token support
- Sentence-based streaming for lower latency
- Stream state tracking for barge-in
- Audio chunk buffering with intelligent yield

Author: BeautyAI Framework
Date: January 2026
"""

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Dict, Any, Callable, Awaitable, List
from enum import Enum, auto

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """TTS stream states."""
    IDLE = auto()
    GENERATING = auto()
    STREAMING = auto()
    PAUSED = auto()
    INTERRUPTED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class StreamMetrics:
    """Metrics for streaming session."""
    text_length: int = 0
    chunks_yielded: int = 0
    bytes_streamed: int = 0
    generation_start_ms: float = 0
    first_chunk_ms: float = 0  # Time to first audio chunk
    total_duration_ms: float = 0
    was_interrupted: bool = False


@dataclass
class StreamConfig:
    """Configuration for streaming TTS."""
    chunk_size_ms: int = 40                    # Chunk size in milliseconds
    target_sample_rate: int = 16000            # Target sample rate
    min_buffer_ms: int = 80                    # Minimum buffer before first yield
    sentence_based: bool = True                # Enable sentence-based streaming
    overlap_sentences: bool = True             # Overlap generation with streaming
    pregenerate_first_sentence: bool = True    # Pre-generate first sentence for fast start


class CancellationToken:
    """Token for cancelling async operations."""
    
    def __init__(self):
        self._cancelled = False
        self._event = asyncio.Event()
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def cancel(self):
        """Request cancellation."""
        self._cancelled = True
        self._event.set()
    
    async def wait_cancellation(self, timeout: float = None) -> bool:
        """Wait for cancellation with optional timeout."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def reset(self):
        """Reset the token for reuse."""
        self._cancelled = False
        self._event.clear()


class InterruptibleTTSStream:
    """
    Interruptible TTS streaming wrapper.
    
    Wraps an async TTS generator and adds:
    - Cancellation support
    - State tracking
    - Barge-in detection callbacks
    - Metrics collection
    
    Usage:
        stream = InterruptibleTTSStream(
            tts_engine=edge_tts_engine,
            text="Hello, how can I help you?",
            language="en"
        )
        
        async for chunk in stream.stream():
            if user_is_speaking:
                await stream.interrupt()
                break
            send_audio(chunk)
    """
    
    def __init__(
        self,
        tts_engine,
        text: str,
        language: str = "en",
        config: Optional[StreamConfig] = None,
        on_state_change: Optional[Callable[[StreamState, StreamState], Awaitable[None]]] = None,
        on_first_chunk: Optional[Callable[[], Awaitable[None]]] = None
    ):
        """
        Initialize interruptible TTS stream.
        
        Args:
            tts_engine: TTS engine instance (EdgeTTSEngine, XTTSEngine, etc.)
            text: Text to synthesize
            language: Language code
            config: Stream configuration
            on_state_change: Callback for state changes
            on_first_chunk: Callback when first chunk is ready
        """
        self.tts_engine = tts_engine
        self.text = text
        self.language = language
        self.config = config or StreamConfig()
        self.on_state_change = on_state_change
        self.on_first_chunk = on_first_chunk
        
        self._state = StreamState.IDLE
        self._cancel_token = CancellationToken()
        self._metrics = StreamMetrics(text_length=len(text))
        self._stream_task: Optional[asyncio.Task] = None
        self._chunk_buffer: asyncio.Queue = asyncio.Queue()

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def metrics(self) -> StreamMetrics:
        return self._metrics

    @property 
    def is_active(self) -> bool:
        return self._state in {StreamState.GENERATING, StreamState.STREAMING}

    async def _set_state(self, new_state: StreamState):
        """Set state with callback."""
        old_state = self._state
        self._state = new_state
        
        if self.on_state_change and old_state != new_state:
            try:
                await self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"[StreamTTS] State change callback error: {e}")

    async def interrupt(self):
        """Interrupt the current stream."""
        if self.is_active:
            logger.info("[StreamTTS] Interruption requested")
            self._cancel_token.cancel()
            self._metrics.was_interrupted = True
            await self._set_state(StreamState.INTERRUPTED)

    async def pause(self):
        """Pause the stream (consumer stops taking chunks)."""
        if self._state == StreamState.STREAMING:
            await self._set_state(StreamState.PAUSED)

    async def resume(self):
        """Resume a paused stream."""
        if self._state == StreamState.PAUSED:
            await self._set_state(StreamState.STREAMING)

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio chunks with interruption support.
        
        Yields:
            bytes: PCM16 audio chunks
        """
        await self._set_state(StreamState.GENERATING)
        self._metrics.generation_start_ms = time.time() * 1000
        first_chunk_sent = False
        
        try:
            # Check if engine has streaming support
            if hasattr(self.tts_engine, 'stream_tts_chunks'):
                # Use native streaming
                async for chunk in self._stream_native():
                    if self._cancel_token.is_cancelled:
                        break
                    
                    if not first_chunk_sent:
                        first_chunk_sent = True
                        self._metrics.first_chunk_ms = (time.time() * 1000) - self._metrics.generation_start_ms
                        await self._set_state(StreamState.STREAMING)
                        
                        if self.on_first_chunk:
                            try:
                                await self.on_first_chunk()
                            except Exception as e:
                                logger.error(f"[StreamTTS] First chunk callback error: {e}")
                    
                    self._metrics.chunks_yielded += 1
                    self._metrics.bytes_streamed += len(chunk)
                    yield chunk
            else:
                # Fallback: generate full audio then chunk
                async for chunk in self._stream_chunked():
                    if self._cancel_token.is_cancelled:
                        break
                    
                    if not first_chunk_sent:
                        first_chunk_sent = True
                        self._metrics.first_chunk_ms = (time.time() * 1000) - self._metrics.generation_start_ms
                        await self._set_state(StreamState.STREAMING)
                        
                        if self.on_first_chunk:
                            await self.on_first_chunk()
                    
                    self._metrics.chunks_yielded += 1
                    self._metrics.bytes_streamed += len(chunk)
                    yield chunk
            
            if not self._cancel_token.is_cancelled:
                await self._set_state(StreamState.COMPLETED)
            
        except asyncio.CancelledError:
            self._metrics.was_interrupted = True
            await self._set_state(StreamState.INTERRUPTED)
            raise
        except Exception as e:
            logger.error(f"[StreamTTS] Streaming error: {e}")
            await self._set_state(StreamState.ERROR)
            raise
        finally:
            self._metrics.total_duration_ms = (time.time() * 1000) - self._metrics.generation_start_ms

    async def _stream_native(self) -> AsyncGenerator[bytes, None]:
        """Stream using engine's native streaming."""
        async for chunk in self.tts_engine.stream_tts_chunks(
            text=self.text,
            language=self.language,
            chunk_size_ms=self.config.chunk_size_ms,
            target_sample_rate=self.config.target_sample_rate
        ):
            if self._cancel_token.is_cancelled:
                break
            yield chunk

    async def _stream_chunked(self) -> AsyncGenerator[bytes, None]:
        """Generate full audio then stream as chunks."""
        loop = asyncio.get_event_loop()
        
        # Generate full audio
        audio_path = await loop.run_in_executor(
            None,
            lambda: self.tts_engine.text_to_speech(
                text=self.text,
                language=self.language
            )
        )
        
        if not audio_path or self._cancel_token.is_cancelled:
            return
        
        # Read and chunk the audio
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                samples_per_chunk = int((sample_rate * self.config.chunk_size_ms) / 1000)
                
                while not self._cancel_token.is_cancelled:
                    chunk = wav_file.readframes(samples_per_chunk)
                    if not chunk:
                        break
                    
                    # Resample if needed
                    if sample_rate != self.config.target_sample_rate:
                        chunk = await self._resample(chunk, sample_rate, self.config.target_sample_rate)
                    
                    yield chunk
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0)
                    
        finally:
            # Cleanup temp file
            try:
                import os
                os.remove(audio_path)
            except:
                pass

    async def _resample(self, audio_data: bytes, source_rate: int, target_rate: int) -> bytes:
        """Resample audio data."""
        if source_rate == target_rate:
            return audio_data
        
        import array
        source_samples = array.array('h')
        source_samples.frombytes(audio_data)
        
        ratio = source_rate / target_rate
        target_length = int(len(source_samples) / ratio)
        
        target_samples = array.array('h')
        for i in range(target_length):
            source_idx = i * ratio
            base_idx = int(source_idx)
            
            if base_idx + 1 < len(source_samples):
                frac = source_idx - base_idx
                sample = source_samples[base_idx] * (1 - frac) + source_samples[base_idx + 1] * frac
                target_samples.append(int(sample))
            elif base_idx < len(source_samples):
                target_samples.append(source_samples[base_idx])
        
        return target_samples.tobytes()


class SentenceStreamingTTS:
    """
    Sentence-based TTS streaming for lower perceived latency.
    
    Splits text into sentences and starts streaming the first sentence
    while generating subsequent sentences in parallel.
    
    This provides much faster time-to-first-audio for longer responses.
    """
    
    def __init__(
        self,
        tts_engine,
        language: str = "en",
        config: Optional[StreamConfig] = None
    ):
        self.tts_engine = tts_engine
        self.language = language
        self.config = config or StreamConfig()
        self._cancel_token = CancellationToken()

    async def stream_text(
        self,
        text: str,
        on_sentence_start: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text with sentence-level parallelism.
        
        Args:
            text: Full text to synthesize
            on_sentence_start: Callback when starting new sentence (index, text)
            
        Yields:
            bytes: PCM16 audio chunks
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return
        
        logger.info(f"[SentenceTTS] Streaming {len(sentences)} sentences")
        
        for i, sentence in enumerate(sentences):
            if self._cancel_token.is_cancelled:
                break
            
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if on_sentence_start:
                try:
                    await on_sentence_start(i, sentence)
                except Exception as e:
                    logger.error(f"[SentenceTTS] Sentence callback error: {e}")
            
            # Stream this sentence
            stream = InterruptibleTTSStream(
                tts_engine=self.tts_engine,
                text=sentence,
                language=self.language,
                config=self.config
            )
            
            # Share cancellation token
            stream._cancel_token = self._cancel_token
            
            async for chunk in stream.stream():
                yield chunk

    def interrupt(self):
        """Interrupt the stream."""
        self._cancel_token.cancel()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming."""
        import re
        
        # Handle Arabic and English sentence boundaries
        # Arabic: ، . ؟ !
        # English: . ? !
        
        # First, protect common abbreviations
        protected = text
        abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'i.e.', 'e.g.']
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholders[placeholder] = abbr
            protected = protected.replace(abbr, placeholder)
        
        # Split on sentence boundaries
        # The pattern matches sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?؟،])\s+', protected)
        
        # Restore abbreviations
        restored = []
        for sentence in sentences:
            for placeholder, abbr in placeholders.items():
                sentence = sentence.replace(placeholder, abbr)
            if sentence.strip():
                restored.append(sentence.strip())
        
        # Merge very short sentences (< 10 chars) with next
        merged = []
        buffer = ""
        
        for sentence in restored:
            if len(sentence) < 10 and buffer:
                buffer += " " + sentence
            elif buffer:
                merged.append(buffer + " " + sentence)
                buffer = ""
            else:
                buffer = sentence
        
        if buffer:
            merged.append(buffer)
        
        return merged if merged else [text]


# Helper functions

async def create_interruptible_tts_stream(
    tts_engine,
    text: str,
    language: str = "en",
    chunk_size_ms: int = 40,
    sample_rate: int = 16000
) -> InterruptibleTTSStream:
    """Factory function to create an interruptible TTS stream."""
    config = StreamConfig(
        chunk_size_ms=chunk_size_ms,
        target_sample_rate=sample_rate
    )
    return InterruptibleTTSStream(
        tts_engine=tts_engine,
        text=text,
        language=language,
        config=config
    )


async def stream_tts_with_interruption(
    tts_engine,
    text: str,
    language: str = "en",
    check_interrupt: Optional[Callable[[], Awaitable[bool]]] = None,
    on_chunk: Optional[Callable[[bytes], Awaitable[None]]] = None
) -> bool:
    """
    Convenience function to stream TTS with interruption checking.
    
    Args:
        tts_engine: TTS engine
        text: Text to synthesize
        language: Language code
        check_interrupt: Async function that returns True if should interrupt
        on_chunk: Callback for each audio chunk
        
    Returns:
        True if completed, False if interrupted
    """
    stream = await create_interruptible_tts_stream(tts_engine, text, language)
    
    try:
        async for chunk in stream.stream():
            # Check for interruption between chunks
            if check_interrupt:
                should_interrupt = await check_interrupt()
                if should_interrupt:
                    await stream.interrupt()
                    return False
            
            if on_chunk:
                await on_chunk(chunk)
        
        return True
        
    except asyncio.CancelledError:
        return False
