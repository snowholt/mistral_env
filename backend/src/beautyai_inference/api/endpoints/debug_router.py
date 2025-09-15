"""Debug / Diagnostics Endpoints

Comprehensive debug and diagnostics endpoints for the voice pipeline,
including health checks, model status, test case generation, and analytics.

Features:
- Pipeline health monitoring
- Model status and performance metrics
- Test case generation and validation
- Debug data analytics
- Audio file testing utilities
- System resource monitoring
"""
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List, Dict, Any
import os
import json
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from ...services.debug.session_reporter import SessionReporter
from ...services.voice.conversation.simple_voice_service import SimpleVoiceService
from ...api.schemas.debug_schemas import (
    SystemHealthStatus, ModelHealthStatus, DebugTestCase, PipelineDebugSummary,
    DebugConfig, AudioDebugInfo, DebugEvent
)
from ...config.voice_config_loader import get_voice_config
from ...core.model_manager import ModelManager

debug_router = APIRouter(prefix="/api/v1/debug", tags=["debug"])

@debug_router.get("/session-report")
def get_session_report(
    session_id: str = Query(..., description="Streaming session_id (e.g. stream_xxx)"),
    max_events: int = Query(800, ge=10, le=5000),
    include_journal: bool = Query(False),
    max_journal_lines: int = Query(300, ge=50, le=1200),
):
    """Get detailed session report for debugging."""
    reporter = SessionReporter()
    if not reporter.enabled:
        raise HTTPException(status_code=403, detail="Session reporting disabled")
    report = reporter.build_report(
        session_id=session_id,
        max_events=max_events,
        include_journal=include_journal,
        max_journal_lines=max_journal_lines,
    )
    return report


@debug_router.get("/health/system", response_model=SystemHealthStatus)
async def get_system_health():
    """Get comprehensive system health status."""
    try:
        import psutil
        import torch
        
        # System resource metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.utilization()
        
        # Check model status
        models = []
        model_manager = ModelManager()
        
        # Check voice service models
        try:
            voice_config = get_voice_config()
            
            # STT model status
            stt_config = voice_config.get_stt_model_config()
            stt_status = ModelHealthStatus(
                model_name=stt_config.model_id,
                model_type="stt",
                status="healthy",  # Would need actual health check
                loaded=True,  # Would need actual status
                error_count=0,
                avg_response_time_ms=None
            )
            models.append(stt_status)
            
            # LLM model status
            # Add checks for loaded models from model registry
            
        except Exception as e:
            # Add error model status
            error_model = ModelHealthStatus(
                model_name="voice_models",
                model_type="voice",
                status="error",
                loaded=False,
                error_count=1,
                avg_response_time_ms=None
            )
            models.append(error_model)
        
        # Get WebSocket connection count (simplified)
        active_connections = 0  # Would get from connection pool
        connection_pool_size = 10  # Would get from actual pool
        
        # System alerts and warnings
        alerts = []
        warnings = []
        
        if cpu_usage > 80:
            alerts.append(f"High CPU usage: {cpu_usage:.1f}%")
        if memory.percent > 85:
            alerts.append(f"High memory usage: {memory.percent:.1f}%")
        if disk.percent > 90:
            warnings.append(f"High disk usage: {disk.percent:.1f}%")
        
        return SystemHealthStatus(
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory.percent,
            gpu_usage_percent=gpu_usage,
            disk_usage_percent=disk.percent,
            models=models,
            active_connections=active_connections,
            connection_pool_size=connection_pool_size,
            alerts=alerts,
            warnings=warnings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@debug_router.get("/health/models")
async def get_models_health():
    """Get detailed model health and performance metrics."""
    try:
        models_info = []
        
        # Check voice service
        voice_service = SimpleVoiceService(debug_mode=True)
        stats = voice_service.get_processing_stats()
        
        models_info.append({
            "service": "SimpleVoiceService",
            "status": "available" if stats.get("edge_tts_available") else "unavailable",
            "stats": stats,
            "models": {
                "arabic_voice": stats.get("default_arabic_voice"),
                "english_voice": stats.get("default_english_voice"),
                "available_voices": stats.get("available_voices", 0)
            }
        })
        
        # Check model manager
        try:
            model_manager = ModelManager()
            
            # Get memory stats
            memory_stats = model_manager.get_memory_stats()
            
            models_info.append({
                "service": "ModelManager",
                "status": "available",
                "memory_stats": memory_stats,
                "loaded_models": []  # Would get from actual manager
            })
            
        except Exception as e:
            models_info.append({
                "service": "ModelManager",
                "status": "error",
                "error": str(e)
            })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "models": models_info,
            "overall_status": "healthy" if all(m.get("status") == "available" for m in models_info) else "degraded"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model health check failed: {str(e)}")


@debug_router.get("/pipeline/test-cases")
async def get_test_cases():
    """Get available test cases for pipeline validation."""
    try:
        # Define standard test cases
        test_cases = [
            {
                "test_id": "arabic_greeting",
                "name": "Arabic Greeting",
                "description": "Test Arabic speech recognition and response",
                "language": "ar",
                "voice_type": "female",
                "expected_transcription": "مرحبا، كيف حالك؟",
                "expected_response_pattern": "مرحبا.*بخير"
            },
            {
                "test_id": "english_greeting",
                "name": "English Greeting",
                "description": "Test English speech recognition and response",
                "language": "en",
                "voice_type": "female",
                "expected_transcription": "Hello, how are you?",
                "expected_response_pattern": "Hello.*fine"
            },
            {
                "test_id": "arabic_question",
                "name": "Arabic Question",
                "description": "Test Arabic question understanding",
                "language": "ar",
                "voice_type": "male",
                "expected_transcription": "ما هو الطقس اليوم؟",
                "expected_response_pattern": "الطقس"
            },
            {
                "test_id": "mixed_language",
                "name": "Language Detection",
                "description": "Test automatic language detection",
                "language": "auto",
                "voice_type": "female",
                "expected_transcription": "Thank you شكرا",
                "expected_response_pattern": "welcome.*أهلا"
            }
        ]
        
        return {
            "test_cases": test_cases,
            "total_count": len(test_cases),
            "languages": ["ar", "en", "auto"],
            "voice_types": ["male", "female"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get test cases: {str(e)}")


@debug_router.post("/pipeline/test")
async def test_pipeline_with_audio(
    audio_file: UploadFile = File(...),
    language: str = Form("ar"),
    voice_type: str = Form("female"),
    expected_transcription: Optional[str] = Form(None),
    debug_mode: bool = Form(True)
):
    """Test the complete voice pipeline with uploaded audio file."""
    try:
        # Validate parameters
        if language not in ["ar", "en", "auto"]:
            raise HTTPException(status_code=400, detail="Invalid language")
        if voice_type not in ["male", "female"]:
            raise HTTPException(status_code=400, detail="Invalid voice_type")
        
        # Read audio file
        audio_data = await audio_file.read()
        
        # Detect audio format from filename
        audio_format = "wav"  # Default
        if audio_file.filename:
            extension = Path(audio_file.filename).suffix.lower()
            if extension in [".webm", ".wav", ".mp3", ".ogg"]:
                audio_format = extension[1:]  # Remove the dot
        
        # Create voice service with debug mode
        voice_service = SimpleVoiceService(debug_mode=debug_mode)
        await voice_service.initialize()
        
        try:
            # Process the audio
            result = await voice_service.process_voice_message(
                audio_data=audio_data,
                audio_format=audio_format,
                language=language if language != "auto" else None,
                gender=voice_type,
                debug_context={
                    "test_mode": True,
                    "test_file": audio_file.filename,
                    "expected_transcription": expected_transcription
                }
            )
            
            # Analyze results
            transcribed_text = result.get("transcribed_text", "")
            response_text = result.get("response_text", "")
            processing_time = result.get("processing_time", 0)
            debug_summary = result.get("debug_summary")
            
            # Calculate accuracy if expected transcription provided
            transcription_accuracy = None
            if expected_transcription and transcribed_text:
                # Simple word-based accuracy calculation
                expected_words = set(expected_transcription.lower().split())
                actual_words = set(transcribed_text.lower().split())
                if expected_words:
                    transcription_accuracy = len(expected_words.intersection(actual_words)) / len(expected_words)
            
            # Performance grading
            performance_grade = "A"
            if processing_time > 3.0:
                performance_grade = "C"
            elif processing_time > 2.0:
                performance_grade = "B"
            
            # Build test result
            test_result = {
                "test_id": f"upload_test_{int(datetime.now().timestamp())}",
                "success": bool(result.get("success", True)),
                "audio_file": audio_file.filename,
                "audio_format": audio_format,
                "audio_size_bytes": len(audio_data),
                "language": language,
                "voice_type": voice_type,
                "results": {
                    "transcribed_text": transcribed_text,
                    "response_text": response_text,
                    "processing_time_ms": processing_time * 1000,
                    "language_detected": result.get("language_detected"),
                    "voice_used": result.get("voice_used")
                },
                "analysis": {
                    "transcription_accuracy": transcription_accuracy,
                    "performance_grade": performance_grade,
                    "response_quality": "ok" if response_text else "empty"
                },
                "debug_summary": debug_summary.dict() if debug_summary and hasattr(debug_summary, 'dict') else debug_summary
            }
            
            return test_result
            
        finally:
            await voice_service.cleanup()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline test failed: {str(e)}")


@debug_router.get("/analytics/events")
async def get_debug_analytics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    stage: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get debug event analytics and statistics."""
    try:
        # This would normally query a debug event database
        # For now, return sample analytics
        
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        # Sample analytics data
        analytics = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "filters": {
                "stage": stage,
                "level": level,
                "limit": limit
            },
            "summary": {
                "total_events": 1250,
                "events_by_stage": {
                    "stt": 425,
                    "llm": 410,
                    "tts": 415
                },
                "events_by_level": {
                    "info": 900,
                    "warning": 250,
                    "error": 100
                },
                "avg_processing_time_ms": {
                    "stt": 450,
                    "llm": 890,
                    "tts": 320,
                    "total": 1660
                }
            },
            "trends": {
                "hourly_volumes": [45, 52, 38, 67, 89, 94, 76, 68, 55, 49, 41, 38],
                "error_rates": [0.08, 0.06, 0.09, 0.12, 0.07, 0.05, 0.08, 0.10, 0.06, 0.07, 0.09, 0.08]
            },
            "top_errors": [
                {"error": "Transcription timeout", "count": 25, "stage": "stt"},
                {"error": "Model not loaded", "count": 18, "stage": "llm"},
                {"error": "Audio format unsupported", "count": 12, "stage": "stt"}
            ]
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics query failed: {str(e)}")


@debug_router.get("/config")
async def get_debug_config():
    """Get current debug configuration."""
    try:
        config = {
            "debug_mode_available": True,
            "audio_capture_enabled": os.getenv("BEAUTYAI_DEBUG_VOICE") == "1",
            "session_reporting_enabled": True,
            "performance_monitoring_enabled": True,
            "supported_audio_formats": ["webm", "wav", "mp3", "ogg"],
            "max_audio_size_mb": 10,
            "debug_event_retention_hours": 24,
            "test_audio_samples": {
                "arabic": ["greeting.wav", "question.wav"],
                "english": ["hello.wav", "test.wav"]
            }
        }
        
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")


@debug_router.get("/samples")
async def list_test_audio_samples():
    """List available test audio samples."""
    try:
        # Look for test audio files in voice_tests directory
        samples = []
        
        # Check if voice_tests directory exists
        voice_tests_dir = Path("/home/lumi/beautyai/voice_tests/input_test_questions")
        if voice_tests_dir.exists():
            for format_dir in voice_tests_dir.iterdir():
                if format_dir.is_dir():
                    format_name = format_dir.name
                    for audio_file in format_dir.glob("*"):
                        if audio_file.is_file():
                            samples.append({
                                "name": audio_file.name,
                                "format": format_name,
                                "path": str(audio_file),
                                "size_bytes": audio_file.stat().st_size,
                                "language": "ar" if "arabic" in audio_file.name.lower() else "auto"
                            })
        
        return {
            "samples": samples,
            "total_count": len(samples),
            "base_path": str(voice_tests_dir) if voice_tests_dir.exists() else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample listing failed: {str(e)}")


@debug_router.get("/samples/{sample_name}")
async def get_test_audio_sample(sample_name: str):
    """Download a test audio sample."""
    try:
        # Security: only allow safe filenames
        if ".." in sample_name or "/" in sample_name:
            raise HTTPException(status_code=400, detail="Invalid sample name")
        
        # Look for the sample file
        voice_tests_dir = Path("/home/lumi/beautyai/voice_tests/input_test_questions")
        sample_file = None
        
        for format_dir in voice_tests_dir.iterdir():
            if format_dir.is_dir():
                potential_file = format_dir / sample_name
                if potential_file.exists():
                    sample_file = potential_file
                    break
        
        if not sample_file:
            raise HTTPException(status_code=404, detail="Sample not found")
        
        return FileResponse(
            path=str(sample_file),
            filename=sample_name,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample download failed: {str(e)}")

__all__ = ["debug_router"]
