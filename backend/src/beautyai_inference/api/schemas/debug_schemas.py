"""
Debug schemas for structured debug data in the voice pipeline.

These schemas provide a standardized format for debugging information,
metrics collection, and performance monitoring across the STT → LLM → TTS pipeline.
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from enum import Enum


class PipelineStage(str, Enum):
    """Pipeline stages for debugging."""
    UPLOAD = "upload"
    STT = "stt" 
    LLM = "llm"
    TTS = "tts"
    COMPLETE = "complete"


class DebugLevel(str, Enum):
    """Debug logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AudioDebugInfo(BaseModel):
    """Debug information for audio processing."""
    duration_ms: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None


class TranscriptionDebugData(BaseModel):
    """Debug data for STT (Speech-to-Text) stage."""
    transcribed_text: str
    language_detected: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_time_ms: float
    model_used: str
    audio_duration_ms: float
    audio_format: str
    audio_info: Optional[AudioDebugInfo] = None
    errors: List[str] = []
    warnings: List[str] = []


class LLMDebugData(BaseModel):
    """Debug data for LLM (Language Model) stage."""
    response_text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    processing_time_ms: float
    model_used: str
    temperature: Optional[float] = None
    thinking_mode: bool = True
    errors: List[str] = []
    warnings: List[str] = []


class TTSDebugData(BaseModel):
    """Debug data for TTS (Text-to-Speech) stage."""
    audio_length_ms: Optional[float] = None
    voice_used: str
    processing_time_ms: float
    output_format: str
    text_length: int
    speech_rate: str = "medium"
    errors: List[str] = []
    warnings: List[str] = []


class DebugEvent(BaseModel):
    """Individual debug event during pipeline processing."""
    stage: str
    level: str
    message: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None


class PipelineDebugSummary(BaseModel):
    """Complete debug summary for a pipeline execution."""
    total_processing_time_ms: float
    transcription_data: Optional[TranscriptionDebugData] = None
    llm_data: Optional[LLMDebugData] = None
    tts_data: Optional[TTSDebugData] = None
    success: bool
    error_message: Optional[str] = None
    debug_events: List[DebugEvent] = []
    stage_timings: Dict[str, float] = {}
    completed_stages: List[str] = []
    bottleneck_stage: Optional[str] = None
    performance_grade: str = "ok"


class WebSocketDebugMessage(BaseModel):
    """Debug message sent via WebSocket during processing."""
    type: str
    stage: str
    level: str
    message: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None


class ModelHealthStatus(BaseModel):
    """Health status of a model."""
    model_name: str
    model_type: str
    status: str
    loaded: bool
    error_count: int
    avg_response_time_ms: Optional[float] = None


class SystemHealthStatus(BaseModel):
    """System health status."""
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    disk_usage_percent: float
    models: List[ModelHealthStatus] = []
    active_connections: int
    connection_pool_size: int
    alerts: List[str] = []
    warnings: List[str] = []


class DebugTestCase(BaseModel):
    """Test case for pipeline validation."""
    test_id: str
    name: str
    description: str
    language: str
    voice_type: str
    expected_transcription: str
    expected_response_pattern: str


class DebugConfig(BaseModel):
    """Debug configuration settings."""
    debug_mode_enabled: bool
    capture_audio: bool
    capture_intermediate_results: bool
    log_level: str
    retention_hours: int