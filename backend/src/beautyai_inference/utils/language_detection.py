"""
Language Detection Utility for BeautyAI Framework.

Provides automatic language detection for text input and intelligent language
matching for voice-to-voice conversations.
"""

import logging
import re
from typing import Optional, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Enumeration of supported languages with their properties."""
    ARABIC = ("ar", "العربية", r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    ENGLISH = ("en", "English", r'[a-zA-Z]')
    SPANISH = ("es", "Español", r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]')
    FRENCH = ("fr", "Français", r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]')
    GERMAN = ("de", "Deutsch", r'[a-zA-ZäöüÄÖÜß]')
    
    def __init__(self, code: str, display_name: str, char_pattern: str):
        self.code = code
        self.display_name = display_name
        self.char_pattern = char_pattern


class LanguageDetector:
    """
    Intelligent language detection utility.
    
    Features:
    - Character-based detection with confidence scoring
    - Text pattern analysis for better accuracy
    - Arabic-optimized detection for beauty industry context
    - Fallback mechanisms for mixed content
    """
    
    def __init__(self):
        self.arabic_keywords = {
            # Beauty and cosmetic terms
            "بوتوكس", "فيلر", "تجميل", "علاج", "عملية", "طبيب", "دكتور", "عيادة",
            "جراحة", "ليزر", "تقشير", "حقن", "تنظيف", "علاج", "استشارة", "موعد",
            # Common greetings and questions
            "مرحبا", "السلام", "أهلا", "كيف", "ماذا", "متى", "أين", "من", "ما", "هل",
            # Medical terms
            "وجه", "جلد", "بشرة", "أنف", "عين", "فم", "شفاه", "جبهة", "خد", "ذقن"
        }
        
        self.english_keywords = {
            # Beauty and cosmetic terms
            "botox", "filler", "cosmetic", "beauty", "treatment", "procedure", "doctor", "clinic",
            "surgery", "laser", "peeling", "injection", "cleaning", "consultation", "appointment",
            # Common greetings and questions
            "hello", "hi", "how", "what", "when", "where", "who", "why", "can", "could", "would",
            # Medical terms
            "face", "skin", "nose", "eye", "mouth", "lips", "forehead", "cheek", "chin"
        }
    
    def detect_language(self, text: str, confidence_threshold: float = 0.3) -> Tuple[str, float]:
        """
        Detect the primary language of the input text.
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence score to consider detection valid
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or not text.strip():
            return "en", 0.0  # Default to English for empty text
        
        text = text.strip()
        logger.debug(f"Detecting language for text: {text[:50]}...")
        
        # Calculate character-based scores
        language_scores = self._calculate_character_scores(text)
        
        # Apply keyword boosting
        language_scores = self._apply_keyword_boosting(text, language_scores)
        
        # Apply contextual boosting for beauty industry
        language_scores = self._apply_contextual_boosting(text, language_scores)
        
        # Find the best match
        best_language = max(language_scores.items(), key=lambda x: x[1])
        detected_language, confidence = best_language
        
        logger.info(f"Language detection result: {detected_language} (confidence: {confidence:.3f})")
        logger.debug(f"All scores: {language_scores}")
        
        # Return result with confidence check
        if confidence >= confidence_threshold:
            return detected_language, confidence
        else:
            # Default to English if confidence is too low
            logger.warning(f"Low confidence ({confidence:.3f}), defaulting to English")
            return "en", confidence
    
    def _calculate_character_scores(self, text: str) -> Dict[str, float]:
        """Calculate language scores based on character patterns."""
        total_chars = len(re.sub(r'\s+', '', text))  # Exclude whitespace
        if total_chars == 0:
            return {lang.code: 0.0 for lang in SupportedLanguage}
        
        scores = {}
        
        for language in SupportedLanguage:
            # Count characters matching this language's pattern
            matches = len(re.findall(language.char_pattern, text))
            score = matches / total_chars
            scores[language.code] = score
        
        return scores
    
    def _apply_keyword_boosting(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply keyword-based boosting to language scores."""
        text_lower = text.lower()
        
        # Arabic keyword boosting
        arabic_matches = sum(1 for keyword in self.arabic_keywords if keyword in text)
        if arabic_matches > 0:
            boost = min(0.3, arabic_matches * 0.1)  # Max boost of 0.3
            scores["ar"] += boost
            logger.debug(f"Arabic keyword boost: {boost} (matches: {arabic_matches})")
        
        # English keyword boosting
        english_matches = sum(1 for keyword in self.english_keywords if keyword in text_lower)
        if english_matches > 0:
            boost = min(0.3, english_matches * 0.1)  # Max boost of 0.3
            scores["en"] += boost
            logger.debug(f"English keyword boost: {boost} (matches: {english_matches})")
        
        return scores
    
    def _apply_contextual_boosting(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply contextual boosting based on text patterns."""
        
        # Arabic contextual patterns
        arabic_patterns = [
            r'أ[نت]ا؟\s+',  # أنا، أنت
            r'هذا|هذه|ذلك|تلك',  # Demonstratives
            r'في\s+|من\s+|إلى\s+',  # Prepositions
            r'[وف]?ال[أ-ي]+',  # Definite article patterns
            r'[\u0621-\u063A\u0641-\u064A]{3,}',  # Arabic word patterns
        ]
        
        for pattern in arabic_patterns:
            if re.search(pattern, text):
                scores["ar"] += 0.1
                logger.debug(f"Arabic pattern match: {pattern}")
        
        # English contextual patterns
        english_patterns = [
            r'\b(the|and|or|but|in|on|at|to|for)\b',  # Common function words
            r'\b[A-Z][a-z]+\b',  # Capitalized words
            r'\b\w+ing\b',  # -ing endings
            r'\b\w+ed\b',   # -ed endings
            r'\b\w+ly\b',   # -ly endings
        ]
        
        for pattern in english_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                scores["en"] += 0.05
                logger.debug(f"English pattern match: {pattern}")
        
        return scores
    
    def is_mixed_language(self, text: str, threshold: float = 0.7) -> bool:
        """
        Check if text contains mixed languages.
        
        Args:
            text: Input text to analyze
            threshold: Threshold below which text is considered mixed
            
        Returns:
            bool: True if text appears to be mixed language
        """
        if not text or not text.strip():
            return False
        
        _, confidence = self.detect_language(text)
        return confidence < threshold
    
    def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a language.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Dict containing language information
        """
        for lang in SupportedLanguage:
            if lang.code == language_code:
                return {
                    "code": lang.code,
                    "name": lang.display_name,
                    "char_pattern": lang.char_pattern,
                    "rtl": lang.code == "ar",  # Right-to-left
                    "whisper_supported": True,
                    "tts_supported": True
                }
        
        # Fallback for unsupported languages
        return {
            "code": language_code,
            "name": language_code.upper(),
            "char_pattern": r'[a-zA-Z]',
            "rtl": False,
            "whisper_supported": True,
            "tts_supported": False
        }
    
    def suggest_response_language(self, input_text: str, 
                                context_history: Optional[list] = None) -> str:
        """
        Suggest the best language for response based on input and context.
        
        Args:
            input_text: User's input text
            context_history: Previous conversation history
            
        Returns:
            str: Suggested language code for response
        """
        # Detect input language
        detected_lang, confidence = self.detect_language(input_text)
        
        # If confidence is high, use detected language
        if confidence >= 0.6:
            logger.info(f"High confidence detection: responding in {detected_lang}")
            return detected_lang
        
        # Check conversation history for consistency
        if context_history:
            history_languages = []
            for msg in context_history[-3:]:  # Check last 3 messages
                if msg.get("role") == "user":
                    hist_lang, hist_conf = self.detect_language(msg.get("content", ""))
                    if hist_conf >= 0.5:
                        history_languages.append(hist_lang)
            
            if history_languages:
                # Use most common language from recent history
                most_common = max(set(history_languages), key=history_languages.count)
                logger.info(f"Using language from conversation history: {most_common}")
                return most_common
        
        # Default fallback logic
        if confidence >= 0.3:
            logger.info(f"Medium confidence detection: responding in {detected_lang}")
            return detected_lang
        else:
            logger.info("Low confidence detection: defaulting to English")
            return "en"


# Global instance for easy import
language_detector = LanguageDetector()


def detect_language(text: str, confidence_threshold: float = 0.3) -> Tuple[str, float]:
    """
    Convenience function for language detection.
    
    Args:
        text: Input text to analyze
        confidence_threshold: Minimum confidence score
        
    Returns:
        Tuple of (language_code, confidence_score)
    """
    return language_detector.detect_language(text, confidence_threshold)


def suggest_response_language(input_text: str, context_history: Optional[list] = None) -> str:
    """
    Convenience function for response language suggestion.
    
    Args:
        input_text: User's input text
        context_history: Previous conversation history
        
    Returns:
        str: Suggested language code for response
    """
    return language_detector.suggest_response_language(input_text, context_history)
