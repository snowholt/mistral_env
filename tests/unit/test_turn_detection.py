"""Unit tests for smart end-of-turn detection.

Tests cover:
- EndOfTurnConfig validation and language-specific defaults
- LinguisticAnalyzer for English and Arabic
- EndOfTurnPredictor confidence scoring
- Turn detection timing and accuracy
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch

# Import modules under test
from beautyai_inference.services.voice.turn_detection.config import EndOfTurnConfig
from beautyai_inference.services.voice.turn_detection.linguistic import LinguisticAnalyzer, LinguisticResult
from beautyai_inference.services.voice.turn_detection.predictor import (
    EndOfTurnPredictor,
    TurnState,
    ConfidenceBreakdown,
)


class TestEndOfTurnConfig:
    """Tests for EndOfTurnConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = EndOfTurnConfig()
        
        assert config.min_silence_ms == 300
        assert config.max_silence_ms == 800
        assert config.confidence_threshold == 0.85
        assert config.poll_interval_ms == 50
        assert config.enabled is True
    
    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        config = EndOfTurnConfig()
        config.linguistic_weight = 0.5
        config.silence_weight = 0.5
        config.asr_stability_weight = 0.5
        config.__post_init__()
        
        total = config.linguistic_weight + config.silence_weight + config.asr_stability_weight
        assert abs(total - 1.0) < 0.01
    
    def test_for_language_arabic(self):
        """Test Arabic-specific configuration."""
        config = EndOfTurnConfig.for_language("ar")
        
        # Arabic should have slightly longer thresholds
        assert config.min_silence_ms >= 300
        assert config.max_silence_ms >= 800
    
    def test_for_language_arabic_sa(self):
        """Test Arabic-SA specific configuration."""
        config = EndOfTurnConfig.for_language("ar-SA")
        
        assert config.min_silence_ms >= 300
    
    def test_for_language_english(self):
        """Test English configuration (defaults)."""
        config = EndOfTurnConfig.for_language("en")
        
        assert config.min_silence_ms == 300
        assert config.max_silence_ms == 800
    
    def test_to_dict(self):
        """Test serialization to dict."""
        config = EndOfTurnConfig()
        result = config.to_dict()
        
        assert "min_silence_ms" in result
        assert "max_silence_ms" in result
        assert "confidence_threshold" in result
        assert "weights" in result
        assert result["weights"]["linguistic"] == config.linguistic_weight
    
    def test_validation_clamps_values(self):
        """Test that validation clamps out-of-range values."""
        config = EndOfTurnConfig()
        config.min_silence_ms = 50  # Below minimum
        config.max_silence_ms = 5000  # Above maximum
        config.confidence_threshold = 1.5  # Above 1.0
        config.__post_init__()
        
        assert config.min_silence_ms >= 100
        assert config.max_silence_ms <= 2000
        assert config.confidence_threshold <= 0.99


class TestLinguisticAnalyzer:
    """Tests for LinguisticAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return LinguisticAnalyzer()
    
    # English tests
    def test_english_terminal_punctuation(self, analyzer):
        """Test detection of sentence-terminal punctuation in English."""
        result = analyzer.analyze("Hello, how are you?", "en")
        assert result.is_complete is True
        assert result.confidence >= 0.9
        assert result.reason == "terminal_punctuation"
    
    def test_english_no_punctuation(self, analyzer):
        """Test incomplete sentence without punctuation."""
        result = analyzer.analyze("I would like to", "en")
        assert result.is_complete is False or result.confidence < 0.9
    
    def test_english_greeting(self, analyzer):
        """Test greeting detection."""
        result = analyzer.analyze("hello", "en")
        assert result.is_complete is True
        assert result.reason in ["greeting_pattern", "short_command"]
    
    def test_english_greeting_with_name(self, analyzer):
        """Test greeting with name."""
        result = analyzer.analyze("Hello there", "en")
        assert result.is_complete is True
    
    def test_english_short_command(self, analyzer):
        """Test short command detection."""
        for cmd in ["yes", "no", "okay", "sure", "thanks"]:
            result = analyzer.analyze(cmd, "en")
            assert result.is_complete is True, f"Failed for: {cmd}"
            assert result.confidence >= 0.8
    
    def test_english_question(self, analyzer):
        """Test question with question mark."""
        result = analyzer.analyze("What time is it?", "en")
        assert result.is_complete is True
        assert result.reason == "terminal_punctuation"
    
    def test_english_incomplete_question(self, analyzer):
        """Test incomplete question without punctuation."""
        result = analyzer.analyze("What time", "en")
        # Should be lower confidence without punctuation
        assert result.confidence < 0.9
    
    # Arabic tests
    def test_arabic_terminal_punctuation(self, analyzer):
        """Test Arabic sentence with punctuation."""
        result = analyzer.analyze("مرحبا، كيف حالك؟", "ar")
        assert result.is_complete is True
        assert result.confidence >= 0.9
    
    def test_arabic_greeting(self, analyzer):
        """Test Arabic greeting."""
        result = analyzer.analyze("مرحبا", "ar")
        assert result.is_complete is True
    
    def test_arabic_short_command(self, analyzer):
        """Test Arabic short commands."""
        for cmd in ["نعم", "لا", "تمام", "شكراً"]:
            result = analyzer.analyze(cmd, "ar")
            assert result.is_complete is True, f"Failed for: {cmd}"
    
    def test_arabic_salam(self, analyzer):
        """Test Arabic greeting - السلام عليكم."""
        result = analyzer.analyze("السلام عليكم", "ar")
        assert result.is_complete is True
    
    # Edge cases
    def test_empty_text(self, analyzer):
        """Test empty text handling."""
        result = analyzer.analyze("", "en")
        assert result.is_complete is False
        assert result.confidence == 0.0
        assert result.reason == "empty_text"
    
    def test_whitespace_only(self, analyzer):
        """Test whitespace-only text."""
        result = analyzer.analyze("   ", "en")
        assert result.is_complete is False
    
    def test_get_completeness_score(self, analyzer):
        """Test convenience method for getting score only."""
        score = analyzer.get_completeness_score("Hello!", "en")
        assert 0 <= score <= 1.0
        assert score > 0.8  # Terminal punctuation should score high


class TestEndOfTurnPredictor:
    """Tests for EndOfTurnPredictor."""
    
    @pytest.fixture
    def predictor(self):
        config = EndOfTurnConfig()
        return EndOfTurnPredictor(config=config, language="en")
    
    @pytest.fixture
    def predictor_ar(self):
        config = EndOfTurnConfig.for_language("ar")
        return EndOfTurnPredictor(config=config, language="ar")
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.config is not None
        assert predictor.language == "en"
        assert predictor.state is not None
    
    def test_on_speech_detected(self, predictor):
        """Test speech detection callback."""
        predictor.on_speech_detected()
        
        assert predictor.state.last_speech_ms is not None
        assert predictor.state.silence_start_ms is None
        assert predictor.state.turn_triggered is False
    
    def test_on_silence_detected(self, predictor):
        """Test silence detection callback."""
        predictor.on_silence_detected()
        
        assert predictor.state.silence_start_ms is not None
    
    def test_silence_duration(self, predictor):
        """Test silence duration calculation."""
        predictor.on_silence_detected()
        time.sleep(0.1)  # 100ms
        
        duration = predictor.get_silence_duration_ms()
        assert duration >= 100  # At least 100ms
    
    def test_update_transcript(self, predictor):
        """Test transcript update."""
        predictor.update_transcript("Hello")
        
        assert predictor.state.current_transcript == "Hello"
        assert len(predictor.state.asr_partials) == 1
    
    def test_transcript_stability(self, predictor):
        """Test ASR stability tracking."""
        predictor.update_transcript("Hello")
        predictor.update_transcript("Hello")
        predictor.update_transcript("Hello")
        
        # Should be stable after 3 identical updates
        assert predictor.state.stable_frame_count >= 2
    
    def test_transcript_instability(self, predictor):
        """Test ASR instability when transcript changes."""
        predictor.update_transcript("Hello")
        predictor.update_transcript("Hello world")
        predictor.update_transcript("Hello world how")
        
        # Should not be marked as stable
        assert predictor.state.stable_frame_count == 0
    
    # Confidence scoring tests
    def test_silence_score_below_min(self, predictor):
        """Test silence score when below minimum threshold."""
        score = predictor.compute_silence_score(100)  # Below 300ms default min
        assert score == 0.0
    
    def test_silence_score_at_min(self, predictor):
        """Test silence score at minimum threshold."""
        score = predictor.compute_silence_score(300)
        assert score >= 0.3
    
    def test_silence_score_at_max(self, predictor):
        """Test silence score at maximum threshold."""
        score = predictor.compute_silence_score(800)
        assert score == 1.0
    
    def test_silence_score_above_max(self, predictor):
        """Test silence score above maximum threshold."""
        score = predictor.compute_silence_score(1000)
        assert score == 1.0
    
    def test_asr_stability_score_empty(self, predictor):
        """Test ASR stability score with no transcript."""
        score = predictor.compute_asr_stability_score()
        assert score == 0.0
    
    def test_asr_stability_score_stable(self, predictor):
        """Test ASR stability score when stable."""
        predictor.update_transcript("Hello")
        predictor.update_transcript("Hello")
        predictor.update_transcript("Hello")
        
        score = predictor.compute_asr_stability_score()
        assert score >= 0.4  # Should be reasonable when stable (2 consecutive frames)
    
    def test_linguistic_score(self, predictor):
        """Test linguistic score computation."""
        predictor.update_transcript("Hello, how are you?")
        
        score = predictor.compute_linguistic_score()
        assert score > 0.5  # Complete sentence should score high
    
    def test_compute_confidence_complete(self, predictor):
        """Test confidence computation for complete turn."""
        # Simulate complete turn: silence + stable ASR + complete sentence
        predictor.on_silence_detected()
        predictor.state.silence_start_ms = predictor._get_current_time_ms() - 700  # 700ms silence
        predictor.update_transcript("Hello!")
        predictor.update_transcript("Hello!")
        predictor.update_transcript("Hello!")
        
        breakdown = predictor.compute_confidence()
        
        assert breakdown.silence_duration_ms >= 700
        assert breakdown.total_confidence > 0.5
        assert breakdown.trigger_reason in ["confidence_threshold", "linguistic_complete", "waiting"]
    
    def test_compute_confidence_max_timeout(self, predictor):
        """Test confidence triggers at max timeout."""
        predictor.on_silence_detected()
        predictor.state.silence_start_ms = predictor._get_current_time_ms() - 1000  # 1000ms silence (> 800ms max)
        predictor.update_transcript("um")  # Incomplete
        
        breakdown = predictor.compute_confidence()
        
        assert breakdown.is_turn_complete is True
        assert breakdown.trigger_reason == "max_silence_timeout"
    
    def test_is_turn_complete_convenience(self, predictor):
        """Test is_turn_complete convenience method."""
        predictor.on_silence_detected()
        predictor.state.silence_start_ms = predictor._get_current_time_ms() - 1000
        
        assert predictor.is_turn_complete() is True
    
    # async tests
    @pytest.mark.asyncio
    async def test_wait_for_turn_end_disabled(self, predictor):
        """Test wait_for_turn_end when smart detection is disabled."""
        predictor.config.enabled = False
        
        start = time.time()
        breakdown = await predictor.wait_for_turn_end()
        elapsed = time.time() - start
        
        assert elapsed >= 0.7  # Should wait max_silence_ms (800ms default)
        assert breakdown.trigger_reason == "disabled_fallback"
    
    @pytest.mark.asyncio
    async def test_wait_for_turn_end_immediate_complete(self, predictor):
        """Test immediate completion when high confidence."""
        predictor.on_silence_detected()
        predictor.state.silence_start_ms = predictor._get_current_time_ms() - 1000
        predictor.update_transcript("Yes.")
        predictor.update_transcript("Yes.")
        predictor.update_transcript("Yes.")
        
        start = time.time()
        breakdown = await predictor.wait_for_turn_end()
        elapsed = time.time() - start
        
        # Should complete quickly (within 1-2 poll intervals)
        assert elapsed < 0.2
        assert breakdown.is_turn_complete is True
    
    @pytest.mark.asyncio
    async def test_wait_for_turn_end_interrupted(self, predictor):
        """Test interruption handling."""
        context = {"interrupted": True}
        
        breakdown = await predictor.wait_for_turn_end(context=context)
        
        assert breakdown.is_turn_complete is False
        assert breakdown.trigger_reason == "interrupted"
    
    @pytest.mark.asyncio
    async def test_wait_for_turn_end_with_callback(self, predictor):
        """Test callback invocation on turn end."""
        callback_called = False
        
        def callback():
            nonlocal callback_called
            callback_called = True
        
        predictor.on_silence_detected()
        predictor.state.silence_start_ms = predictor._get_current_time_ms() - 1000
        
        await predictor.wait_for_turn_end(callback=callback)
        
        assert callback_called is True
    
    # Metrics tests
    def test_get_metrics(self, predictor):
        """Test metrics retrieval."""
        metrics = predictor.get_metrics()
        
        assert "turn_count" in metrics
        assert "avg_detection_time_ms" in metrics
        assert "config" in metrics
        assert "language" in metrics
    
    def test_reset(self, predictor):
        """Test state reset."""
        predictor.on_silence_detected()
        predictor.update_transcript("Hello")
        
        predictor.reset()
        
        assert predictor.state.silence_start_ms is None
        assert predictor.state.current_transcript == ""
        assert len(predictor.state.asr_partials) == 0
    
    def test_reset_metrics(self, predictor):
        """Test metrics reset."""
        predictor._turn_count = 10
        predictor._total_detection_time_ms = 5000
        
        predictor.reset_metrics()
        
        assert predictor._turn_count == 0
        assert predictor._total_detection_time_ms == 0


class TestArabicTurnDetection:
    """Test Arabic-specific turn detection scenarios."""
    
    @pytest.fixture
    def predictor(self):
        config = EndOfTurnConfig.for_language("ar")
        return EndOfTurnPredictor(config=config, language="ar")
    
    def test_arabic_greeting_complete(self, predictor):
        """Test Arabic greeting is detected as complete."""
        predictor.on_silence_detected()
        predictor.state.silence_start_ms = predictor._get_current_time_ms() - 400
        predictor.update_transcript("مرحبا")
        predictor.update_transcript("مرحبا")
        predictor.update_transcript("مرحبا")
        
        breakdown = predictor.compute_confidence()
        
        # Arabic greeting should have high linguistic score
        assert breakdown.linguistic_score > 0.5
    
    def test_arabic_question_complete(self, predictor):
        """Test Arabic question with punctuation."""
        predictor.on_silence_detected()
        predictor.state.silence_start_ms = predictor._get_current_time_ms() - 500
        predictor.update_transcript("ما هو الوقت؟")
        predictor.update_transcript("ما هو الوقت؟")
        predictor.update_transcript("ما هو الوقت؟")
        
        breakdown = predictor.compute_confidence()
        
        assert breakdown.linguistic_score > 0.8  # Terminal punctuation


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def predictor(self):
        config = EndOfTurnConfig()
        return EndOfTurnPredictor(config=config, language="en")
    
    def test_silence_duration_no_silence(self, predictor):
        """Test silence duration when silence hasn't started."""
        duration = predictor.get_silence_duration_ms()
        assert duration == 0.0
    
    def test_update_empty_transcript(self, predictor):
        """Test updating with empty transcript."""
        predictor.update_transcript("")
        
        assert predictor.state.current_transcript == ""
    
    def test_confidence_no_silence_no_transcript(self, predictor):
        """Test confidence with no data."""
        breakdown = predictor.compute_confidence()
        
        assert breakdown.total_confidence < 0.5
        assert breakdown.is_turn_complete is False
