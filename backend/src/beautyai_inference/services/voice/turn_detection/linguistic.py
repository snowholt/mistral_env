"""Linguistic analysis for end-of-turn detection.

Provides sentence completeness detection for Arabic and English,
analyzing whether a transcript appears to be a complete utterance.
"""

import re
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LinguisticResult:
    """Result of linguistic analysis."""
    is_complete: bool
    confidence: float
    reason: str
    language: str


class LinguisticAnalyzer:
    """Analyzes text for linguistic completeness.
    
    Supports Arabic and English with language-specific patterns for:
    - Sentence-terminal punctuation
    - Short command detection (yes/no/ok)
    - Question detection
    - Greeting detection
    
    This is a heuristic-based analyzer optimized for conversational speech.
    """
    
    # Sentence-terminal punctuation by language
    TERMINAL_PUNCTUATION = {
        "en": r"[.!?]$",
        "ar": r"[.!?؟،]$",  # Arabic question mark and comma (often sentence-final)
    }
    
    # Short affirmative/negative commands by language
    SHORT_COMMANDS = {
        "en": {
            "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", 
            "right", "fine", "great", "good", "thanks", "bye", "hello",
            "hi", "hey", "please", "stop", "wait", "go", "next", "done",
        },
        "ar": {
            "نعم", "لا", "أيوه", "إيه", "طيب", "حسناً", "تمام", "شكراً",
            "مرحبا", "أهلاً", "السلام عليكم", "وعليكم السلام",
            "من فضلك", "يلّا", "خلاص", "كفى", "بس",
        },
    }
    
    # Question words that suggest incomplete questions if at the end
    QUESTION_STARTERS = {
        "en": {"what", "who", "where", "when", "why", "how", "which", "can", "could", "would", "will", "do", "does", "is", "are"},
        "ar": {"ما", "ماذا", "من", "أين", "متى", "لماذا", "كيف", "هل", "أي"},
    }
    
    # Greeting patterns (complete on their own)
    GREETING_PATTERNS = {
        "en": [
            r"^(hello|hi|hey|good\s+(morning|afternoon|evening))(\s+there)?[.!]?$",
            r"^(bye|goodbye|see\s+you|take\s+care)[.!]?$",
        ],
        "ar": [
            r"^(مرحبا|أهلاً|السلام\s*عليكم|صباح\s+الخير|مساء\s+الخير)[.!]?$",
            r"^(وعليكم\s*السلام|مع\s+السلامة)[.!]?$",
        ],
    }
    
    def __init__(self):
        """Initialize the analyzer."""
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        for lang, patterns in self.GREETING_PATTERNS.items():
            self._compiled_patterns[f"greeting_{lang}"] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        for lang, pattern in self.TERMINAL_PUNCTUATION.items():
            self._compiled_patterns[f"terminal_{lang}"] = re.compile(pattern)
    
    def _get_language_key(self, language: str) -> str:
        """Normalize language code to 'en' or 'ar'."""
        if language.startswith("ar"):
            return "ar"
        return "en"
    
    def analyze(self, text: str, language: str = "en") -> LinguisticResult:
        """Analyze text for linguistic completeness.
        
        Args:
            text: The transcript text to analyze
            language: Language code (en, ar, en-US, ar-SA, etc.)
        
        Returns:
            LinguisticResult with completeness assessment
        """
        if not text or not text.strip():
            return LinguisticResult(
                is_complete=False,
                confidence=0.0,
                reason="empty_text",
                language=language,
            )
        
        text = text.strip()
        lang_key = self._get_language_key(language)
        
        # Check for terminal punctuation (highest confidence)
        if self._has_terminal_punctuation(text, lang_key):
            return LinguisticResult(
                is_complete=True,
                confidence=0.95,
                reason="terminal_punctuation",
                language=language,
            )
        
        # Check for greeting patterns (complete on their own)
        if self._is_greeting(text, lang_key):
            return LinguisticResult(
                is_complete=True,
                confidence=0.90,
                reason="greeting_pattern",
                language=language,
            )
        
        # Check for short commands (likely complete)
        if self._is_short_command(text, lang_key):
            return LinguisticResult(
                is_complete=True,
                confidence=0.85,
                reason="short_command",
                language=language,
            )
        
        # Check for structural completeness hints
        structural_confidence = self._analyze_structure(text, lang_key)
        if structural_confidence > 0.7:
            return LinguisticResult(
                is_complete=True,
                confidence=structural_confidence,
                reason="structural_analysis",
                language=language,
            )
        
        # Default: uncertain, let other signals decide
        return LinguisticResult(
            is_complete=False,
            confidence=0.3,
            reason="uncertain",
            language=language,
        )
    
    def _has_terminal_punctuation(self, text: str, lang_key: str) -> bool:
        """Check if text ends with sentence-terminal punctuation."""
        pattern = self._compiled_patterns.get(f"terminal_{lang_key}")
        if pattern:
            return bool(pattern.search(text))
        return text.rstrip()[-1] in ".!?" if text.rstrip() else False
    
    def _is_greeting(self, text: str, lang_key: str) -> bool:
        """Check if text matches a greeting pattern."""
        patterns = self._compiled_patterns.get(f"greeting_{lang_key}", [])
        text_clean = text.strip().lower()
        return any(p.match(text_clean) for p in patterns)
    
    def _is_short_command(self, text: str, lang_key: str) -> bool:
        """Check if text is a short command/response."""
        text_clean = text.strip().lower()
        words = text_clean.split()
        
        if len(words) > 3:
            return False
        
        commands = self.SHORT_COMMANDS.get(lang_key, set())
        # Check if any word matches a command
        return any(word.strip(".,!?") in commands for word in words)
    
    def _analyze_structure(self, text: str, lang_key: str) -> float:
        """Analyze sentence structure for completeness hints.
        
        Returns a confidence score between 0 and 1.
        """
        words = text.split()
        word_count = len(words)
        
        # Very short text is likely incomplete unless it's a command
        if word_count < 2:
            return 0.2
        
        # Medium length sentences are more likely complete
        if 3 <= word_count <= 10:
            confidence = 0.5
            
            # Boost if it starts with capital (English) or looks like a statement
            if lang_key == "en" and text[0].isupper():
                confidence += 0.1
            
            # Boost if no trailing question word
            last_word = words[-1].lower().strip(".,!?")
            question_starters = self.QUESTION_STARTERS.get(lang_key, set())
            if last_word not in question_starters:
                confidence += 0.15
            
            return min(confidence, 0.8)
        
        # Long sentences: likely complete or waiting for more
        if word_count > 10:
            return 0.6
        
        return 0.4
    
    def get_completeness_score(self, text: str, language: str = "en") -> float:
        """Get just the completeness confidence score.
        
        Convenience method for the predictor.
        """
        result = self.analyze(text, language)
        return result.confidence if result.is_complete else result.confidence * 0.5
