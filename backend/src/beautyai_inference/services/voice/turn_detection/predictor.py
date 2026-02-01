"""End-of-turn predictor with multi-signal confidence scoring.

This module implements the core prediction logic for detecting when a user
has finished speaking using a weighted combination of:
- Silence duration (how long since last speech)
- ASR stability (are transcription results stable)
- Linguistic completeness (does the text look like a complete utterance)

The predictor is designed to replace fixed-timeout approaches with adaptive
confidence-based turn detection that can trigger early when confident.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

from .config import EndOfTurnConfig
from .linguistic import LinguisticAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """Tracks the state of the current turn for prediction."""
    
    # Timing
    silence_start_ms: Optional[float] = None
    last_speech_ms: Optional[float] = None
    
    # ASR tracking
    asr_partials: deque = field(default_factory=lambda: deque(maxlen=5))
    last_stable_transcript: str = ""
    stable_frame_count: int = 0
    
    # Current transcript
    current_transcript: str = ""
    
    # State flags
    is_processing: bool = False
    turn_triggered: bool = False


@dataclass  
class ConfidenceBreakdown:
    """Detailed breakdown of confidence scoring for debugging."""
    
    silence_score: float
    asr_stability_score: float
    linguistic_score: float
    total_confidence: float
    silence_duration_ms: float
    is_turn_complete: bool
    trigger_reason: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "silence_score": round(self.silence_score, 3),
            "asr_stability_score": round(self.asr_stability_score, 3),
            "linguistic_score": round(self.linguistic_score, 3),
            "total_confidence": round(self.total_confidence, 3),
            "silence_duration_ms": round(self.silence_duration_ms, 1),
            "is_turn_complete": self.is_turn_complete,
            "trigger_reason": self.trigger_reason,
        }


class EndOfTurnPredictor:
    """Predicts end-of-turn using multi-signal confidence scoring.
    
    Usage:
        predictor = EndOfTurnPredictor(config, language="ar")
        
        # During voice processing
        predictor.on_speech_detected()  # When VAD detects speech
        predictor.on_silence_detected()  # When VAD detects silence
        predictor.update_transcript("partial transcript")  # From ASR
        
        # Wait for turn end
        await predictor.wait_for_turn_end(callback=on_turn_end)
    """
    
    def __init__(
        self,
        config: Optional[EndOfTurnConfig] = None,
        language: str = "en",
    ):
        """Initialize the predictor.
        
        Args:
            config: Configuration for turn detection. If None, uses defaults.
            language: Language code for linguistic analysis.
        """
        self.config = config or EndOfTurnConfig.for_language(language)
        self.language = language
        self.linguistic_analyzer = LinguisticAnalyzer()
        
        # State tracking
        self.state = TurnState()
        self._reset_state()
        
        # Metrics
        self._turn_count = 0
        self._total_detection_time_ms = 0
    
    def _reset_state(self):
        """Reset state for a new turn."""
        self.state = TurnState()
        self.state.asr_partials = deque(maxlen=self.config.asr_partial_history_size)
    
    def _get_current_time_ms(self) -> float:
        """Get current time in milliseconds."""
        return time.time() * 1000
    
    def on_speech_detected(self):
        """Called when VAD detects speech.
        
        Resets silence timer and updates state.
        """
        now = self._get_current_time_ms()
        self.state.last_speech_ms = now
        self.state.silence_start_ms = None
        self.state.turn_triggered = False
        
        logger.debug(f"[TurnPredict] Speech detected at {now:.0f}ms")
    
    def on_silence_detected(self):
        """Called when VAD detects silence.
        
        Starts the silence timer if not already started.
        """
        if self.state.silence_start_ms is None:
            self.state.silence_start_ms = self._get_current_time_ms()
            logger.debug(f"[TurnPredict] Silence started at {self.state.silence_start_ms:.0f}ms")
    
    def update_transcript(self, transcript: str):
        """Update the current transcript from ASR.
        
        Tracks stability by comparing consecutive partials.
        """
        if not transcript:
            return
        
        self.state.current_transcript = transcript.strip()
        self.state.asr_partials.append(self.state.current_transcript)
        
        # Check stability
        if len(self.state.asr_partials) >= 2:
            if self.state.asr_partials[-1] == self.state.asr_partials[-2]:
                self.state.stable_frame_count += 1
                self.state.last_stable_transcript = self.state.current_transcript
            else:
                self.state.stable_frame_count = 0
        
        logger.debug(
            f"[TurnPredict] Transcript updated: "
            f"'{transcript[:30]}...' stable_frames={self.state.stable_frame_count}"
        )
    
    def get_silence_duration_ms(self) -> float:
        """Get current silence duration in milliseconds."""
        if self.state.silence_start_ms is None:
            return 0.0
        return self._get_current_time_ms() - self.state.silence_start_ms
    
    def compute_silence_score(self, silence_ms: float) -> float:
        """Compute confidence score from silence duration.
        
        Uses a sigmoid-like curve:
        - 0-min_silence: 0.0
        - min_silence-max_silence: linear ramp 0.3 to 0.9
        - >max_silence: 1.0
        """
        min_ms = self.config.min_silence_ms
        max_ms = self.config.max_silence_ms
        
        if silence_ms < min_ms:
            return 0.0
        
        if silence_ms >= max_ms:
            return 1.0
        
        # Linear interpolation between min and max
        ratio = (silence_ms - min_ms) / (max_ms - min_ms)
        # Scale to 0.3-0.9 range
        return 0.3 + (ratio * 0.6)
    
    def compute_asr_stability_score(self) -> float:
        """Compute confidence score from ASR stability.
        
        Higher score if transcript has been stable for multiple frames.
        """
        if not self.state.current_transcript:
            return 0.0
        
        stable_frames = self.state.stable_frame_count
        required_frames = self.config.asr_stability_frames
        
        if stable_frames >= required_frames:
            return 1.0
        elif stable_frames > 0:
            return stable_frames / required_frames * 0.7
        else:
            return 0.1  # Still processing, low confidence
    
    def compute_linguistic_score(self) -> float:
        """Compute confidence score from linguistic analysis."""
        if not self.state.current_transcript:
            return 0.0
        
        return self.linguistic_analyzer.get_completeness_score(
            self.state.current_transcript, 
            self.language
        )
    
    def compute_confidence(self) -> ConfidenceBreakdown:
        """Compute overall confidence that turn is complete.
        
        Returns a breakdown of all component scores and the final decision.
        """
        silence_ms = self.get_silence_duration_ms()
        
        # Compute individual scores
        silence_score = self.compute_silence_score(silence_ms)
        asr_score = self.compute_asr_stability_score()
        linguistic_score = self.compute_linguistic_score()
        
        # Weighted combination
        total = (
            silence_score * self.config.silence_weight +
            asr_score * self.config.asr_stability_weight +
            linguistic_score * self.config.linguistic_weight
        )
        
        # Determine if turn is complete
        is_complete = False
        trigger_reason = "waiting"
        
        # Check threshold
        if total >= self.config.confidence_threshold:
            is_complete = True
            trigger_reason = "confidence_threshold"
        
        # Force trigger at max silence
        elif silence_ms >= self.config.max_silence_ms:
            is_complete = True
            trigger_reason = "max_silence_timeout"
        
        # Special case: very high linguistic confidence with some silence
        elif linguistic_score > 0.9 and silence_ms >= self.config.min_silence_ms:
            is_complete = True
            trigger_reason = "linguistic_complete"
        
        return ConfidenceBreakdown(
            silence_score=silence_score,
            asr_stability_score=asr_score,
            linguistic_score=linguistic_score,
            total_confidence=total,
            silence_duration_ms=silence_ms,
            is_turn_complete=is_complete,
            trigger_reason=trigger_reason,
        )
    
    def is_turn_complete(self) -> bool:
        """Quick check if turn appears complete.
        
        For use in tight loops where full breakdown isn't needed.
        """
        return self.compute_confidence().is_turn_complete
    
    async def wait_for_turn_end(
        self,
        callback: Optional[Callable[[], Any]] = None,
        context: Optional[Dict] = None,
    ) -> ConfidenceBreakdown:
        """Wait for turn to complete, polling at configured interval.
        
        This is the main entry point for turn detection. It polls the
        confidence score until either:
        - Confidence exceeds threshold (early trigger)
        - Maximum silence duration exceeded (forced trigger)
        - Cancellation requested
        
        Args:
            callback: Optional callback to invoke when turn ends
            context: Optional context dict for interrupt checking
        
        Returns:
            ConfidenceBreakdown with final decision details
        """
        if not self.config.enabled:
            # Fallback to simple timeout
            logger.info("[TurnPredict] Smart detection disabled, using fixed timeout")
            await asyncio.sleep(self.config.max_silence_ms / 1000)
            return ConfidenceBreakdown(
                silence_score=1.0,
                asr_stability_score=0.5,
                linguistic_score=0.5,
                total_confidence=0.67,
                silence_duration_ms=self.config.max_silence_ms,
                is_turn_complete=True,
                trigger_reason="disabled_fallback",
            )
        
        start_time = self._get_current_time_ms()
        poll_interval = self.config.poll_interval_ms / 1000  # Convert to seconds
        
        self.state.is_processing = True
        last_log_time = 0
        
        try:
            while True:
                # Check for interruption
                if context and context.get("interrupted", False):
                    logger.info("[TurnPredict] Interrupted, aborting wait")
                    self.state.turn_triggered = False
                    return ConfidenceBreakdown(
                        silence_score=0.0,
                        asr_stability_score=0.0,
                        linguistic_score=0.0,
                        total_confidence=0.0,
                        silence_duration_ms=0.0,
                        is_turn_complete=False,
                        trigger_reason="interrupted",
                    )
                
                # Compute confidence
                breakdown = self.compute_confidence()
                
                # Log periodically (every 200ms)
                now = self._get_current_time_ms()
                if now - last_log_time > 200:
                    logger.debug(
                        f"[TurnPredict] Poll: conf={breakdown.total_confidence:.2f} "
                        f"silence={breakdown.silence_duration_ms:.0f}ms "
                        f"reason={breakdown.trigger_reason}"
                    )
                    last_log_time = now
                
                # Check if turn is complete
                if breakdown.is_turn_complete:
                    detection_time = now - start_time
                    self._turn_count += 1
                    self._total_detection_time_ms += detection_time
                    
                    logger.info(
                        f"[TurnPredict] ✅ Turn complete! "
                        f"reason={breakdown.trigger_reason} "
                        f"conf={breakdown.total_confidence:.2f} "
                        f"detection_time={detection_time:.0f}ms"
                    )
                    
                    self.state.turn_triggered = True
                    
                    if callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback()
                        else:
                            callback()
                    
                    return breakdown
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
        
        except asyncio.CancelledError:
            logger.debug("[TurnPredict] Wait cancelled")
            raise
        
        finally:
            self.state.is_processing = False
    
    def get_metrics(self) -> dict:
        """Get metrics for monitoring/debugging."""
        avg_detection_time = (
            self._total_detection_time_ms / self._turn_count 
            if self._turn_count > 0 else 0
        )
        
        return {
            "turn_count": self._turn_count,
            "avg_detection_time_ms": round(avg_detection_time, 1),
            "config": self.config.to_dict(),
            "language": self.language,
        }
    
    def reset(self):
        """Reset predictor state for a new conversation."""
        self._reset_state()
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self._turn_count = 0
        self._total_detection_time_ms = 0
