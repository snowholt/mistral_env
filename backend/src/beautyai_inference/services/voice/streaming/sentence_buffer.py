"""Sentence streaming buffer for progressive TTS.

This module provides utilities for detecting sentence boundaries in
streaming LLM output and queuing sentences for TTS synthesis.

The key insight: instead of waiting for complete LLM response, we
detect sentence boundaries and send completed sentences to TTS
immediately, achieving much lower perceived latency.
"""

import asyncio
import re
import logging
from typing import AsyncGenerator, Optional, List, Callable, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SentenceStreamConfig:
    """Configuration for sentence streaming.
    
    Attributes:
        min_sentence_chars: Minimum characters for a valid sentence.
            Prevents streaming very short fragments.
        max_sentence_chars: Maximum characters before forced sentence break.
            Prevents very long sentences that delay TTS.
        sentence_end_patterns: Regex patterns for sentence boundaries.
        flush_on_newline: Whether to treat newlines as sentence boundaries.
        language: Language code for language-specific patterns.
    """
    
    min_sentence_chars: int = 10
    max_sentence_chars: int = 200
    flush_on_newline: bool = True
    language: str = "en"
    
    # Sentence boundary patterns by language
    SENTENCE_PATTERNS = {
        "en": r'[.!?]+\s*',
        "ar": r'[.!?؟،]+\s*',  # Arabic includes ؟ (question mark) and ، (comma)
    }
    
    def get_pattern(self) -> str:
        """Get sentence boundary pattern for configured language."""
        lang_key = "ar" if self.language.startswith("ar") else "en"
        return self.SENTENCE_PATTERNS.get(lang_key, self.SENTENCE_PATTERNS["en"])


@dataclass
class StreamedSentence:
    """A single sentence extracted from streaming text."""
    
    text: str
    index: int
    is_final: bool = False
    forced_break: bool = False  # True if broken due to max_chars


class SentenceStreamBuffer:
    """Buffer for accumulating text and yielding complete sentences.
    
    Usage:
        buffer = SentenceStreamBuffer(config)
        
        async for sentence in buffer.process_tokens(token_generator):
            # Each sentence is yielded as soon as boundary is detected
            await tts.synthesize(sentence.text)
        
        # Or use feed() for manual control:
        for token in tokens:
            for sentence in buffer.feed(token):
                await tts.synthesize(sentence.text)
        
        # Flush remaining text at end
        for sentence in buffer.flush():
            await tts.synthesize(sentence.text)
    """
    
    def __init__(self, config: Optional[SentenceStreamConfig] = None):
        """Initialize the buffer.
        
        Args:
            config: Configuration for sentence detection.
        """
        self.config = config or SentenceStreamConfig()
        self._buffer = ""
        self._sentence_index = 0
        self._pattern = re.compile(self.config.get_pattern())
        
        # Metrics
        self._total_chars = 0
        self._sentences_yielded = 0
    
    def feed(self, text: str) -> List[StreamedSentence]:
        """Feed text into buffer and extract complete sentences.
        
        Args:
            text: New text fragment (token or chunk) to process.
        
        Returns:
            List of complete sentences extracted (may be empty).
        """
        if not text:
            return []
        
        self._buffer += text
        self._total_chars += len(text)
        
        sentences = []
        
        # Check for sentence boundaries
        while True:
            sentence = self._extract_next_sentence()
            if sentence:
                sentences.append(sentence)
                self._sentences_yielded += 1
            else:
                break
        
        return sentences
    
    def _extract_next_sentence(self) -> Optional[StreamedSentence]:
        """Try to extract the next complete sentence from buffer."""
        if not self._buffer:
            return None
        
        # Check for forced break (max chars exceeded)
        if len(self._buffer) >= self.config.max_sentence_chars:
            # Find a reasonable break point (space, comma, or forced)
            break_point = self._find_break_point(self.config.max_sentence_chars)
            sentence_text = self._buffer[:break_point].strip()
            self._buffer = self._buffer[break_point:].lstrip()
            
            if len(sentence_text) >= self.config.min_sentence_chars:
                sentence = StreamedSentence(
                    text=sentence_text,
                    index=self._sentence_index,
                    forced_break=True,
                )
                self._sentence_index += 1
                return sentence
        
        # Check for natural sentence boundary
        match = self._pattern.search(self._buffer)
        if match:
            end_pos = match.end()
            sentence_text = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:].lstrip()
            
            if len(sentence_text) >= self.config.min_sentence_chars:
                sentence = StreamedSentence(
                    text=sentence_text,
                    index=self._sentence_index,
                )
                self._sentence_index += 1
                return sentence
        
        # Check for newline break (if enabled)
        if self.config.flush_on_newline and '\n' in self._buffer:
            newline_pos = self._buffer.index('\n')
            sentence_text = self._buffer[:newline_pos].strip()
            self._buffer = self._buffer[newline_pos + 1:].lstrip()
            
            if len(sentence_text) >= self.config.min_sentence_chars:
                sentence = StreamedSentence(
                    text=sentence_text,
                    index=self._sentence_index,
                )
                self._sentence_index += 1
                return sentence
        
        return None
    
    def _find_break_point(self, max_pos: int) -> int:
        """Find a reasonable break point before max_pos.
        
        Prefers word boundaries (spaces) then punctuation.
        """
        text = self._buffer[:max_pos]
        
        # Try to find last space
        last_space = text.rfind(' ')
        if last_space > self.config.min_sentence_chars:
            return last_space
        
        # Try to find last comma or other break character
        for char in [',', '،', ';', '-', ':']:
            last_break = text.rfind(char)
            if last_break > self.config.min_sentence_chars:
                return last_break + 1
        
        # Force break at max_pos
        return max_pos
    
    def flush(self) -> List[StreamedSentence]:
        """Flush any remaining text as final sentence(s).
        
        Call this when the stream is complete to get any trailing text.
        
        Returns:
            List of remaining sentences (may be empty if buffer empty).
        """
        sentences = []
        
        while self._buffer.strip():
            text = self._buffer.strip()
            
            # If very short, just yield it
            if len(text) < self.config.max_sentence_chars:
                if len(text) >= self.config.min_sentence_chars:
                    sentences.append(StreamedSentence(
                        text=text,
                        index=self._sentence_index,
                        is_final=True,
                    ))
                    self._sentence_index += 1
                self._buffer = ""
                break
            
            # Otherwise, extract sentences until empty
            sentence = self._extract_next_sentence()
            if sentence:
                sentences.append(sentence)
            else:
                # Can't extract more, yield remainder
                if len(text) >= 3:  # Minimum for final flush
                    sentences.append(StreamedSentence(
                        text=text,
                        index=self._sentence_index,
                        is_final=True,
                    ))
                    self._sentence_index += 1
                self._buffer = ""
                break
        
        # Mark last sentence as final
        if sentences:
            sentences[-1].is_final = True
        
        return sentences
    
    async def process_tokens(
        self, 
        token_generator: AsyncGenerator[str, None],
        on_sentence: Optional[Callable[[StreamedSentence], Any]] = None,
    ) -> AsyncGenerator[StreamedSentence, None]:
        """Process a stream of tokens and yield complete sentences.
        
        Args:
            token_generator: Async generator yielding text tokens.
            on_sentence: Optional callback for each sentence.
        
        Yields:
            StreamedSentence objects as sentences are detected.
        """
        async for token in token_generator:
            sentences = self.feed(token)
            for sentence in sentences:
                if on_sentence:
                    result = on_sentence(sentence)
                    if asyncio.iscoroutine(result):
                        await result
                yield sentence
        
        # Flush remaining
        for sentence in self.flush():
            if on_sentence:
                result = on_sentence(sentence)
                if asyncio.iscoroutine(result):
                    await result
            yield sentence
    
    def reset(self):
        """Reset buffer state for new stream."""
        self._buffer = ""
        self._sentence_index = 0
        self._total_chars = 0
        self._sentences_yielded = 0
    
    def get_metrics(self) -> dict:
        """Get buffer metrics."""
        return {
            "total_chars": self._total_chars,
            "sentences_yielded": self._sentences_yielded,
            "buffer_remaining": len(self._buffer),
        }


class TTSStreamQueue:
    """Queue for managing progressive TTS synthesis.
    
    Handles:
    - Queuing sentences for TTS synthesis
    - Concurrent TTS generation (synthesize next while playing current)
    - Interruption handling (clear queue on user speech)
    - Audio chunk streaming
    """
    
    def __init__(self, tts_engine, language: str = "en", max_concurrent: int = 2):
        """Initialize the TTS queue.
        
        Args:
            tts_engine: TTS engine instance (EdgeTTSEngine, etc.)
            language: Language code for TTS.
            max_concurrent: Maximum concurrent TTS generations.
        """
        self.tts_engine = tts_engine
        self.language = language
        self.max_concurrent = max_concurrent
        
        self._sentence_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._interrupted = False
        self._processing = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._sentences_processed = 0
        self._total_tts_time_ms = 0
    
    async def start(self):
        """Start the TTS processing worker."""
        if self._worker_task is None:
            self._processing = True
            self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self):
        """Stop the TTS processing worker."""
        self._processing = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
    
    def interrupt(self):
        """Signal interruption - clears queues."""
        self._interrupted = True
        # Clear queues
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    def reset(self):
        """Reset for new conversation."""
        self._interrupted = False
        self._sentences_processed = 0
        self._total_tts_time_ms = 0
    
    async def queue_sentence(self, sentence: StreamedSentence):
        """Add a sentence to the TTS queue.
        
        Args:
            sentence: Sentence to synthesize.
        """
        if not self._interrupted:
            await self._sentence_queue.put(sentence)
    
    async def get_audio_chunk(self) -> Optional[dict]:
        """Get next audio chunk from the queue.
        
        Returns:
            Dict with audio data, or None if queue is empty.
        """
        try:
            return await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    async def _worker_loop(self):
        """Worker loop that processes sentences and generates TTS."""
        import time
        
        while self._processing:
            try:
                # Get next sentence
                sentence = await asyncio.wait_for(
                    self._sentence_queue.get(), 
                    timeout=0.5
                )
                
                if self._interrupted:
                    continue
                
                # Generate TTS
                start_time = time.time()
                
                try:
                    tts_type = type(self.tts_engine).__name__
                    
                    if tts_type == "EdgeTTSEngine":
                        # Use streaming method
                        audio_chunks = []
                        async for chunk in self.tts_engine.stream_tts_chunks(
                            text=sentence.text,
                            language=self.language,
                            gender="female",
                        ):
                            if self._interrupted:
                                break
                            audio_chunks.append(chunk)
                        
                        if audio_chunks and not self._interrupted:
                            # Combine chunks for this sentence
                            combined = b''.join(audio_chunks)
                            await self._audio_queue.put({
                                "audio_pcm": combined,
                                "sentence_index": sentence.index,
                                "is_final": sentence.is_final,
                                "text": sentence.text,
                            })
                    else:
                        # Fallback: use text_to_speech and read file
                        import os
                        audio_path = self.tts_engine.text_to_speech(
                            text=sentence.text,
                            language=self.language,
                        )
                        
                        if audio_path and os.path.exists(audio_path) and not self._interrupted:
                            with open(audio_path, 'rb') as f:
                                audio_data = f.read()
                            
                            await self._audio_queue.put({
                                "audio_wav": audio_data,
                                "sentence_index": sentence.index,
                                "is_final": sentence.is_final,
                                "text": sentence.text,
                            })
                            
                            try:
                                os.remove(audio_path)
                            except:
                                pass
                    
                    tts_time = (time.time() - start_time) * 1000
                    self._total_tts_time_ms += tts_time
                    self._sentences_processed += 1
                    
                    logger.debug(
                        f"[TTSQueue] Processed sentence {sentence.index} "
                        f"({len(sentence.text)} chars) in {tts_time:.0f}ms"
                    )
                    
                except Exception as e:
                    logger.error(f"[TTSQueue] TTS error for sentence {sentence.index}: {e}")
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[TTSQueue] Worker error: {e}")
    
    def get_metrics(self) -> dict:
        """Get queue metrics."""
        avg_time = (
            self._total_tts_time_ms / self._sentences_processed 
            if self._sentences_processed > 0 else 0
        )
        return {
            "sentences_processed": self._sentences_processed,
            "avg_tts_time_ms": round(avg_time, 1),
            "pending_sentences": self._sentence_queue.qsize(),
            "pending_audio": self._audio_queue.qsize(),
        }
