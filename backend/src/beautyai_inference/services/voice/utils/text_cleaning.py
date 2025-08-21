"""Text cleaning utilities for voice services.

Provides functions to:
 - Remove LLM "thinking" blocks (<think>...</think>)
 - Strip emojis / pictographs that cause awkward TTS verbalization
 - Produce a final sanitized string for TTS

These utilities are intentionally lightweight (pure regex) and avoid
any heavy NLP dependencies to keep latency minimal in real-time voice
pipelines.
"""
from __future__ import annotations

import re
from typing import Final

__all__ = [
    "remove_thinking_blocks",
    "strip_emojis",
    "sanitize_tts_text",
]

_THINKING_PATTERN: Final[re.Pattern] = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Emoji / pictograph unicode ranges consolidated. We deliberately keep Arabic letters,
# Latin letters, digits, punctuation and diacritics untouched.
_EMOJI_PATTERN: Final[re.Pattern] = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric extended
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"
    "\uFE0F"                  # variation selectors
    "]+"
)

def remove_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> sections and collapse whitespace."""
    if not text:
        return text
    cleaned = _THINKING_PATTERN.sub("", text)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

def strip_emojis(text: str) -> str:
    """Remove emojis & pictographs that TTS engines would verbalize awkwardly."""
    if not text:
        return text
    no_emoji = _EMOJI_PATTERN.sub("", text)
    no_emoji = re.sub(r"\s{2,}", " ", no_emoji)
    return no_emoji.strip()

def sanitize_tts_text(text: str) -> str:
    """Sanitization pipeline for TTS: only remove thinking blocks."""
    return remove_thinking_blocks(text)
