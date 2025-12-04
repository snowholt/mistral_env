"""
Transcription Cleaner Utility for BeautyAI Framework.

This module provides utilities to detect and clean repetition patterns from 
Whisper transcription output. This is particularly important for Arabic language
where Whisper occasionally produces repeated word patterns.

Common patterns detected:
- Single word repetition: "مرحبا مرحبا مرحبا مرحبا" → "مرحبا"
- Phrase repetition: "كيف حالك كيف حالك كيف حالك" → "كيف حالك"
- End-of-sentence repetition: "هذا جيد جيد جيد" → "هذا جيد"
- Mixed good + bad transcription: "مرحبا كيف حالك مرحبا مرحبا مرحبا" → "مرحبا كيف حالك"

Author: BeautyAI Framework
Date: December 2025
"""

import re
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """Result of transcription cleaning."""
    original_text: str
    cleaned_text: str
    was_cleaned: bool
    repetition_detected: bool
    pattern_type: Optional[str] = None
    confidence: float = 1.0


def detect_word_repetition(text: str, min_repeats: int = 3) -> Tuple[bool, Optional[str], int]:
    """
    Detect if a word or short phrase is repeated consecutively.
    
    Args:
        text: The text to analyze
        min_repeats: Minimum number of repetitions to trigger detection
        
    Returns:
        Tuple of (is_repetition, repeated_unit, count)
    """
    if not text or len(text.strip()) < 3:
        return False, None, 0
    
    words = text.strip().split()
    if len(words) < min_repeats:
        return False, None, 0
    
    # Check for single word repetition (most common case)
    # e.g., "hello hello hello hello" or "مرحبا مرحبا مرحبا مرحبا"
    first_word = words[0]
    if all(word == first_word for word in words):
        return True, first_word, len(words)
    
    # Check for 2-word phrase repetition
    # e.g., "how are how are how are" or "كيف حالك كيف حالك كيف حالك"
    if len(words) >= 4 and len(words) % 2 == 0:
        phrase = f"{words[0]} {words[1]}"
        is_phrase_repeat = True
        for i in range(0, len(words), 2):
            if f"{words[i]} {words[i+1]}" != phrase:
                is_phrase_repeat = False
                break
        if is_phrase_repeat:
            return True, phrase, len(words) // 2
    
    # Check for 3-word phrase repetition
    if len(words) >= 6 and len(words) % 3 == 0:
        phrase = f"{words[0]} {words[1]} {words[2]}"
        is_phrase_repeat = True
        for i in range(0, len(words), 3):
            if f"{words[i]} {words[i+1]} {words[i+2]}" != phrase:
                is_phrase_repeat = False
                break
        if is_phrase_repeat:
            return True, phrase, len(words) // 3
    
    return False, None, 0


def detect_tail_repetition(text: str, min_repeats: int = 3) -> Tuple[bool, str, str]:
    """
    Detect if the text ends with repeated words (good content + repetition tail).
    
    Args:
        text: The text to analyze
        min_repeats: Minimum number of tail repetitions
        
    Returns:
        Tuple of (has_tail_repetition, clean_prefix, repeated_tail)
    """
    if not text or len(text.strip()) < 5:
        return False, text, ""
    
    words = text.strip().split()
    if len(words) < min_repeats + 1:
        return False, text, ""
    
    # Check if the last N words are the same
    last_word = words[-1]
    repeat_count = 0
    
    for i in range(len(words) - 1, -1, -1):
        if words[i] == last_word:
            repeat_count += 1
        else:
            break
    
    if repeat_count >= min_repeats:
        # We have tail repetition
        clean_words = words[:len(words) - repeat_count + 1]  # Keep one instance
        return True, " ".join(clean_words), last_word
    
    # Check for 2-word tail repetition
    if len(words) >= 4:
        last_phrase = f"{words[-2]} {words[-1]}"
        phrase_count = 0
        
        for i in range(len(words) - 2, -1, -2):
            if i >= 1 and f"{words[i-1]} {words[i]}" == last_phrase:
                phrase_count += 1
            else:
                break
        
        if phrase_count >= min_repeats:
            clean_words = words[:len(words) - (phrase_count * 2) + 2]
            return True, " ".join(clean_words), last_phrase
    
    return False, text, ""


def detect_arabic_repetition_patterns(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect common Arabic repetition patterns that Whisper produces.
    
    Common patterns in Arabic:
    - Filler words repeated: "يعني يعني يعني"
    - Common phrases: "أنا أنا أنا", "هذا هذا هذا"
    - Transcription artifacts: random repeated syllables
    
    Args:
        text: The text to analyze
        
    Returns:
        Tuple of (is_problematic_pattern, cleaned_text)
    """
    if not text:
        return False, None
    
    # Common Arabic filler words that get repeated
    arabic_fillers = [
        "يعني", "هذا", "أنا", "هو", "هي", "نعم", "لا", "أه", "آه",
        "طيب", "خلاص", "تمام", "ماشي", "أوكي", "حسنا"
    ]
    
    words = text.strip().split()
    
    # Check if text is mostly one repeated word
    if len(words) >= 3:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If one word makes up more than 70% of the text
        most_common = max(word_counts.items(), key=lambda x: x[1])
        if most_common[1] / len(words) > 0.7:
            return True, most_common[0]
    
    return False, None


def clean_transcription(
    text: str, 
    language: str = "ar",
    min_repeats: int = 3,
    aggressive: bool = False
) -> CleaningResult:
    """
    Clean transcription output by removing repetition patterns.
    
    This is the main entry point for the transcription cleaner.
    
    Args:
        text: Raw transcription from Whisper
        language: Language code (ar, en, etc.)
        min_repeats: Minimum repetitions to trigger cleaning
        aggressive: If True, apply more aggressive cleaning for problematic audio
        
    Returns:
        CleaningResult with original and cleaned text
    """
    if not text or not text.strip():
        return CleaningResult(
            original_text=text or "",
            cleaned_text="",
            was_cleaned=False,
            repetition_detected=False
        )
    
    original = text.strip()
    cleaned = original
    pattern_type = None
    repetition_detected = False
    
    # Step 1: Check for full repetition (entire text is repeated words)
    is_full_repeat, repeated_unit, count = detect_word_repetition(cleaned, min_repeats)
    if is_full_repeat:
        logger.info(f"[TRANSCRIPTION-CLEANER] Full repetition detected: '{repeated_unit}' x{count}")
        cleaned = repeated_unit
        pattern_type = "full_repetition"
        repetition_detected = True
    
    # Step 2: Check for tail repetition (good content + repeated tail)
    if not repetition_detected:
        has_tail, clean_prefix, tail = detect_tail_repetition(cleaned, min_repeats)
        if has_tail:
            logger.info(f"[TRANSCRIPTION-CLEANER] Tail repetition detected: '{tail}' removed from end")
            cleaned = clean_prefix
            pattern_type = "tail_repetition"
            repetition_detected = True
    
    # Step 3: Arabic-specific patterns
    if not repetition_detected and language in ["ar", "arabic"]:
        is_arabic_pattern, arabic_clean = detect_arabic_repetition_patterns(cleaned)
        if is_arabic_pattern and arabic_clean:
            logger.info(f"[TRANSCRIPTION-CLEANER] Arabic repetition pattern detected")
            cleaned = arabic_clean
            pattern_type = "arabic_pattern"
            repetition_detected = True
    
    # Step 4: Remove excessive whitespace
    cleaned = " ".join(cleaned.split())
    
    # Step 5: Filter out very short nonsense (aggressive mode)
    if aggressive and len(cleaned) < 3 and repetition_detected:
        logger.info(f"[TRANSCRIPTION-CLEANER] Aggressive mode: discarding short result '{cleaned}'")
        cleaned = ""
    
    was_cleaned = cleaned != original
    
    if was_cleaned:
        logger.info(f"[TRANSCRIPTION-CLEANER] Cleaned: '{original[:50]}...' → '{cleaned[:50]}...'")
    
    return CleaningResult(
        original_text=original,
        cleaned_text=cleaned,
        was_cleaned=was_cleaned,
        repetition_detected=repetition_detected,
        pattern_type=pattern_type,
        confidence=0.9 if repetition_detected else 1.0
    )


def is_valid_transcription(text: str, min_length: int = 2) -> bool:
    """
    Check if a transcription is valid and meaningful.
    
    Args:
        text: The transcription to validate
        min_length: Minimum length in characters
        
    Returns:
        True if valid, False if likely garbage
    """
    if not text or not text.strip():
        return False
    
    clean = text.strip()
    
    # Too short
    if len(clean) < min_length:
        return False
    
    # Single repeated character
    if len(set(clean.replace(" ", ""))) == 1:
        return False
    
    return True


# Convenience function for quick filtering
def filter_whisper_output(text: str, language: str = "ar") -> str:
    """
    Quick filter for Whisper output - returns cleaned text or empty string.
    
    This is the recommended function to use in production pipelines.
    
    Args:
        text: Raw Whisper transcription
        language: Language code
        
    Returns:
        Cleaned text or empty string if invalid
    """
    result = clean_transcription(text, language=language)
    
    if not is_valid_transcription(result.cleaned_text):
        return ""
    
    return result.cleaned_text
