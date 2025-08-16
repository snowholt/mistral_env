import pytest

from beautyai_inference.services.voice.utils.text_cleaning import (
    remove_thinking_blocks,
    strip_emojis,
    sanitize_tts_text,
)


def test_remove_thinking_blocks_basic():
    raw = "Hello <think>internal reasoning</think>world"
    assert remove_thinking_blocks(raw) == "Hello world"


def test_remove_thinking_blocks_multiline():
    raw = "Before <think>multi\nline\n reasoning</think> After"
    cleaned = remove_thinking_blocks(raw)
    assert "think" not in cleaned.lower()
    assert cleaned.startswith("Before") and cleaned.endswith("After")


def test_strip_emojis_preserves_arabic():
    text = "مرحبا 😊 كيف الحال؟"
    cleaned = strip_emojis(text)
    assert "😊" not in cleaned
    # Arabic letters remain
    assert "مرحبا" in cleaned


def test_strip_emojis_multiple():
    text = "Hi 🙂🙂 there 🚀!"
    cleaned = strip_emojis(text)
    assert "🙂" not in cleaned and "🚀" not in cleaned
    assert "Hi" in cleaned and "there" in cleaned


def test_sanitize_combined():
    text = "<think>reasoning🙂</think> Result 😀"
    sanitized = sanitize_tts_text(text)
    assert "think" not in sanitized.lower()
    assert "🙂" not in sanitized and "😀" not in sanitized
    assert "Result" in sanitized


def test_idempotent():
    text = "Plain text"
    assert sanitize_tts_text(text) == text
