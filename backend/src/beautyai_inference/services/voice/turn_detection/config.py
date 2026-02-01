"""Configuration for smart end-of-turn detection.

Environment variables:
- VOICE_TURN_MIN_SILENCE_MS: Minimum silence before considering turn end (default: 300)
- VOICE_TURN_MAX_SILENCE_MS: Maximum silence before forcing turn end (default: 800)
- VOICE_TURN_CONFIDENCE_THRESHOLD: Confidence threshold for early trigger (default: 0.85)
- VOICE_SMART_TURN_DETECTION: Enable smart turn detection (default: 1)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EndOfTurnConfig:
    """Configuration for end-of-turn prediction.
    
    The confidence scoring uses weighted combination of:
    - Silence duration (how long user has been quiet)
    - ASR stability (are transcription results stable)
    - Linguistic completeness (does text look like a complete utterance)
    
    Attributes:
        min_silence_ms: Absolute minimum silence before considering turn end.
            Below this, confidence is 0 regardless of other signals.
        max_silence_ms: Safety cap - force turn end after this duration.
            Prevents indefinite waiting even if confidence never reaches threshold.
        confidence_threshold: Minimum confidence score to trigger turn end early.
            Higher = more conservative, lower = faster but more false positives.
        poll_interval_ms: How often to check confidence during silence.
        
        # Signal weights (must sum to 1.0)
        linguistic_weight: Weight for linguistic completeness signal.
        silence_weight: Weight for silence duration signal.
        asr_stability_weight: Weight for ASR token stability signal.
        
        # ASR stability settings
        asr_stability_frames: Number of consecutive identical ASR results
            required to consider transcription "stable".
        asr_partial_history_size: How many ASR partial results to track.
        
        # Linguistic settings
        short_utterance_bonus: Extra confidence for short commands (yes/no/ok).
        short_utterance_max_words: Max words to be considered "short utterance".
        
        # Language-specific
        supported_languages: Languages with linguistic analysis support.
    """
    
    # Core timing parameters
    min_silence_ms: int = field(default_factory=lambda: int(
        os.getenv("VOICE_TURN_MIN_SILENCE_MS", "300")
    ))
    max_silence_ms: int = field(default_factory=lambda: int(
        os.getenv("VOICE_TURN_MAX_SILENCE_MS", "800")
    ))
    confidence_threshold: float = field(default_factory=lambda: float(
        os.getenv("VOICE_TURN_CONFIDENCE_THRESHOLD", "0.85")
    ))
    poll_interval_ms: int = 50
    
    # Signal weights (sum to 1.0)
    linguistic_weight: float = 0.40
    silence_weight: float = 0.35
    asr_stability_weight: float = 0.25
    
    # ASR stability
    asr_stability_frames: int = 3
    asr_partial_history_size: int = 5
    
    # Linguistic analysis
    short_utterance_bonus: float = 0.20
    short_utterance_max_words: int = 3
    
    # Supported languages for linguistic analysis
    supported_languages: tuple = ("ar", "en", "ar-SA", "en-US")
    
    # Feature flag
    enabled: bool = field(default_factory=lambda: 
        os.getenv("VOICE_SMART_TURN_DETECTION", "1") == "1"
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure weights sum to 1.0
        total_weight = self.linguistic_weight + self.silence_weight + self.asr_stability_weight
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.linguistic_weight /= total_weight
            self.silence_weight /= total_weight
            self.asr_stability_weight /= total_weight
        
        # Validate ranges
        self.min_silence_ms = max(100, min(1000, self.min_silence_ms))
        self.max_silence_ms = max(self.min_silence_ms + 100, min(2000, self.max_silence_ms))
        self.confidence_threshold = max(0.5, min(0.99, self.confidence_threshold))
    
    @classmethod
    def for_language(cls, language: str) -> "EndOfTurnConfig":
        """Get language-optimized configuration.
        
        Arabic tends to have longer pauses between phrases, so we use
        slightly longer silence thresholds.
        """
        config = cls()
        
        if language.startswith("ar"):
            # Arabic: slightly more patience for natural pauses
            config.min_silence_ms = max(config.min_silence_ms, 350)
            config.max_silence_ms = max(config.max_silence_ms, 900)
        
        return config
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "min_silence_ms": self.min_silence_ms,
            "max_silence_ms": self.max_silence_ms,
            "confidence_threshold": self.confidence_threshold,
            "poll_interval_ms": self.poll_interval_ms,
            "weights": {
                "linguistic": self.linguistic_weight,
                "silence": self.silence_weight,
                "asr_stability": self.asr_stability_weight,
            },
            "enabled": self.enabled,
        }
