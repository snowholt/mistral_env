"""Turn detection service for smart end-of-turn prediction.

This module implements ML-based confidence scoring to detect when a user
has finished speaking, replacing fixed-timeout approaches with adaptive
multi-signal fusion.

Key components:
- EndOfTurnConfig: Configuration for turn detection parameters
- EndOfTurnPredictor: Main predictor class with confidence scoring
- LinguisticAnalyzer: Sentence completeness detection for Arabic/English
"""

from .config import EndOfTurnConfig
from .predictor import EndOfTurnPredictor
from .linguistic import LinguisticAnalyzer

__all__ = [
    "EndOfTurnConfig",
    "EndOfTurnPredictor", 
    "LinguisticAnalyzer",
]
