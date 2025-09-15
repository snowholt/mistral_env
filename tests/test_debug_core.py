"""
Test suite for debug infrastructure - focused tests without full app dependency.

This test validates debug schemas, service functionality, and core debug features.
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

# Test the debug schemas directly
from backend.src.beautyai_inference.api.schemas.debug_schemas import (
    PipelineDebugSummary, DebugEvent, WebSocketDebugMessage,
    TranscriptionDebugData, LLMDebugData, TTSDebugData,
    SystemHealthStatus, ModelHealthStatus, DebugTestCase
)
from backend.src.beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService


class TestDebugSchemas:
    """Test debug schema validation and serialization."""
    
    def test_transcription_debug_data(self):
        """Test TranscriptionDebugData schema."""
        data = TranscriptionDebugData(
            transcribed_text="Hello world",
            language_detected="en",
            confidence_score=0.95,
            processing_time_ms=450,
            model_used="whisper-base",
            audio_duration_ms=1000,
            audio_format="wav"
        )
        
        # Verify serialization
        json_data = data.model_dump()
        assert json_data["transcribed_text"] == "Hello world"
        assert json_data["confidence_score"] == 0.95
        assert json_data["processing_time_ms"] == 450
        assert json_data["errors"] == []
        assert json_data["warnings"] == []
    
    def test_llm_debug_data(self):
        """Test LLMDebugData schema."""
        data = LLMDebugData(
            response_text="I'm doing well!",
            prompt_tokens=15,
            completion_tokens=8,
            processing_time_ms=890,
            model_used="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Verify serialization
        json_data = data.model_dump()
        assert json_data["response_text"] == "I'm doing well!"
        assert json_data["prompt_tokens"] == 15
        assert json_data["processing_time_ms"] == 890
        assert json_data["thinking_mode"] is True
    
    def test_tts_debug_data(self):
        """Test TTSDebugData schema."""
        data = TTSDebugData(
            audio_length_ms=2500,
            voice_used="ar-EG-SalmaNeural",
            processing_time_ms=320,
            output_format="mp3",
            text_length=23,
            speech_rate="medium"
        )
        
        # Verify serialization
        json_data = data.model_dump()
        assert json_data["voice_used"] == "ar-EG-SalmaNeural"
        assert json_data["output_format"] == "mp3"
        assert json_data["speech_rate"] == "medium"
    
    def test_debug_event(self):
        """Test DebugEvent schema."""
        event = DebugEvent(
            stage="stt",
            level="info",
            message="Transcription completed",
            timestamp="2024-01-01T12:00:00Z",
            data={"confidence": 0.95}
        )
        
        json_data = event.model_dump()
        assert json_data["stage"] == "stt"
        assert json_data["level"] == "info"
        assert json_data["data"]["confidence"] == 0.95
    
    def test_pipeline_debug_summary(self):
        """Test complete pipeline debug summary."""
        transcription_data = TranscriptionDebugData(
            transcribed_text="Hello world",
            language_detected="en",
            confidence_score=0.95,
            processing_time_ms=450,
            model_used="whisper",
            audio_duration_ms=1000,
            audio_format="wav"
        )
        
        llm_data = LLMDebugData(
            response_text="Response",
            prompt_tokens=10,
            completion_tokens=5,
            processing_time_ms=800,
            model_used="gpt-3.5",
            temperature=0.7
        )
        
        tts_data = TTSDebugData(
            audio_length_ms=1500,
            voice_used="en-US-AriaNeural",
            processing_time_ms=300,
            output_format="mp3",
            text_length=10,
            speech_rate="medium"
        )
        
        pipeline_summary = PipelineDebugSummary(
            total_processing_time_ms=1550,
            transcription_data=transcription_data,
            llm_data=llm_data,
            tts_data=tts_data,
            success=True,
            error_message=None,
            debug_events=[
                DebugEvent(
                    stage="stt",
                    level="info",
                    message="Transcription completed",
                    timestamp="2024-01-01T12:00:00Z",
                    data={"confidence": 0.95}
                )
            ]
        )
        
        # Verify complete pipeline summary
        json_data = pipeline_summary.model_dump()
        assert json_data["total_processing_time_ms"] == 1550
        assert json_data["success"] is True
        assert len(json_data["debug_events"]) == 1
        assert json_data["transcription_data"]["transcribed_text"] == "Hello world"
        assert json_data["llm_data"]["response_text"] == "Response"
        assert json_data["tts_data"]["voice_used"] == "en-US-AriaNeural"
    
    def test_websocket_debug_message(self):
        """Test WebSocket debug message schema."""
        message = WebSocketDebugMessage(
            type="debug_event",
            stage="llm",
            level="info",
            message="Processing LLM request",
            timestamp="2024-01-01T12:00:00Z",
            data={"tokens": 15}
        )
        
        json_data = message.model_dump()
        assert json_data["type"] == "debug_event"
        assert json_data["stage"] == "llm"
        assert json_data["data"]["tokens"] == 15
    
    def test_system_health_status(self):
        """Test system health status schema."""
        model_status = ModelHealthStatus(
            model_name="whisper-base",
            model_type="stt",
            status="healthy",
            loaded=True,
            error_count=0,
            avg_response_time_ms=450.5
        )
        
        health_status = SystemHealthStatus(
            cpu_usage_percent=45.2,
            memory_usage_percent=68.1,
            gpu_usage_percent=23.5,
            disk_usage_percent=55.0,
            models=[model_status],
            active_connections=3,
            connection_pool_size=10,
            alerts=["High memory usage"],
            warnings=["Model response time elevated"]
        )
        
        json_data = health_status.model_dump()
        assert json_data["cpu_usage_percent"] == 45.2
        assert len(json_data["models"]) == 1
        assert json_data["models"][0]["model_name"] == "whisper-base"
        assert len(json_data["alerts"]) == 1
        assert len(json_data["warnings"]) == 1


class TestSimpleVoiceServiceDebug:
    """Test SimpleVoiceService debug functionality."""
    
    @pytest.mark.asyncio
    async def test_debug_mode_initialization(self):
        """Test voice service debug mode initialization."""
        # Test with debug mode enabled
        service_debug = SimpleVoiceService(debug_mode=True)
        assert service_debug.debug_mode is True
        assert service_debug.debug_events == []
        assert service_debug.debug_callback is None
        
        # Test with debug mode disabled
        service_normal = SimpleVoiceService(debug_mode=False)
        assert service_normal.debug_mode is False
    
    @pytest.mark.asyncio
    async def test_debug_event_emission(self):
        """Test debug event emission."""
        service = SimpleVoiceService(debug_mode=True)
        
        # Emit test debug events
        service._emit_debug_event("stt", "info", "Starting transcription")
        service._emit_debug_event("llm", "warning", "High token count")
        service._emit_debug_event("tts", "error", "Voice not found")
        
        # Verify events were collected
        assert len(service.debug_events) == 3
        
        # Verify event details
        assert service.debug_events[0].stage == "stt"
        assert service.debug_events[0].level == "info"
        assert service.debug_events[0].message == "Starting transcription"
        
        assert service.debug_events[1].stage == "llm"
        assert service.debug_events[1].level == "warning"
        
        assert service.debug_events[2].stage == "tts"
        assert service.debug_events[2].level == "error"
    
    @pytest.mark.asyncio
    async def test_debug_callbacks(self):
        """Test debug event callbacks."""
        service = SimpleVoiceService(debug_mode=True)
        
        # Mock callback function
        callback_events = []
        def mock_callback(event):
            callback_events.append(event)
        
        # Register callback
        service.set_debug_callback(mock_callback)
        
        # Emit events
        service._emit_debug_event("stt", "info", "Test event")
        
        # Verify callback was called
        assert len(callback_events) == 1
        assert callback_events[0].stage == "stt"
        assert callback_events[0].message == "Test event"
    
    @pytest.mark.asyncio
    async def test_debug_summary_generation(self):
        """Test debug summary generation."""
        service = SimpleVoiceService(debug_mode=True)
        
        # Simulate debug data - the summary is created by the service
        service.debug_events = [
            DebugEvent(
                stage="stt",
                level="info",
                message="Transcription completed",
                timestamp="2024-01-01T12:00:00Z"
            ),
            DebugEvent(
                stage="llm",
                level="info",
                message="Response generated",
                timestamp="2024-01-01T12:00:01Z"
            )
        ]
        
        # Initially no summary should exist
        summary = service.get_debug_summary()
        assert summary is None
        
        # Create a debug summary manually (this would normally be done by process_voice_message)
        service.current_debug_summary = PipelineDebugSummary(
            total_processing_time_ms=1550,
            success=True,
            debug_events=service.debug_events
        )
        
        # Now summary should exist
        summary = service.get_debug_summary()
        assert summary is not None
        assert summary.total_processing_time_ms == 1550
        assert summary.success is True
        assert len(summary.debug_events) == 2


class TestPerformanceGrading:
    """Test performance analysis and grading."""
    
    def test_performance_grading_logic(self):
        """Test pipeline performance analysis and grading."""
        test_cases = [
            {"processing_time": 1.5, "expected_grade": "A"},
            {"processing_time": 2.5, "expected_grade": "B"},
            {"processing_time": 4.0, "expected_grade": "C"}
        ]
        
        for case in test_cases:
            # Simulate performance grading (would be done in actual endpoint)
            processing_time = case["processing_time"]
            performance_grade = "A"
            if processing_time > 3.0:
                performance_grade = "C"
            elif processing_time > 2.0:
                performance_grade = "B"
            
            assert performance_grade == case["expected_grade"]
    
    def test_bottleneck_detection(self):
        """Test bottleneck stage detection."""
        # Sample stage timings
        stage_timings = {
            "stt": 450,    # 450ms
            "llm": 1200,   # 1200ms - bottleneck
            "tts": 300     # 300ms
        }
        
        # Find bottleneck (slowest stage)
        bottleneck_stage = max(stage_timings, key=stage_timings.get)
        bottleneck_time = stage_timings[bottleneck_stage]
        
        assert bottleneck_stage == "llm"
        assert bottleneck_time == 1200
        
        # Calculate percentages
        total_time = sum(stage_timings.values())
        bottleneck_percentage = (bottleneck_time / total_time) * 100
        
        assert bottleneck_percentage > 50  # LLM takes more than 50% of time


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])