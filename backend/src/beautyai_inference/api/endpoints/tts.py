"""
TTS (Text-to-Speech) Test Endpoint

Provides a simple API for testing TTS engines from the WebRTC debug UI.

Endpoints:
- POST /api/v1/tts/generate - Generate speech from text
- GET /api/v1/tts/engines - List available TTS engines

Author: BeautyAI Framework
Date: 2026-01-30
"""

import logging
import os
import time
import tempfile
import base64
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


class TTSGenerateRequest(BaseModel):
    """Request model for TTS generation."""
    text: str
    engine: str = "edge-tts"  # edge-tts, saudi-tts, chatterbox-multilingual
    language: str = "ar"
    speaker_wav: Optional[str] = None  # Optional custom speaker reference
    exaggeration: Optional[float] = 0.5  # Chatterbox only
    cfg_weight: Optional[float] = 0.5   # Chatterbox only


class TTSGenerateResponse(BaseModel):
    """Response model for TTS generation."""
    success: bool
    engine: str
    language: str
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None
    generation_time_ms: Optional[float] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None


# Store generated audio files temporarily
_tts_output_dir = Path(tempfile.gettempdir()) / "beautyai_tts_test"
_tts_output_dir.mkdir(exist_ok=True)

# Engine instances (lazy loaded)
_engine_instances = {}


def _get_edge_tts_engine():
    """Get or create Edge TTS engine."""
    if "edge-tts" not in _engine_instances:
        from ...inference_engines.voice.tts import EdgeTTSEngine
        from ...config.config_manager import ModelConfig
        
        config = ModelConfig(name="edge-tts-test", model_id="edge-tts", engine_type="edge_tts")
        engine = EdgeTTSEngine(config)
        _engine_instances["edge-tts"] = engine
        logger.info("✅ Edge TTS engine loaded for testing")
    
    return _engine_instances["edge-tts"]


def _get_chatterbox_engine():
    """Get Chatterbox engine via ModelManager."""
    if "chatterbox-multilingual" not in _engine_instances:
        try:
            from ...core.model_manager import get_model_manager
            
            manager = get_model_manager()
            engine = manager.get_tts_engine("chatterbox-multilingual")
            
            if engine is None:
                raise RuntimeError("ModelManager returned None for chatterbox-multilingual")
            
            _engine_instances["chatterbox-multilingual"] = engine
            logger.info("✅ Chatterbox Multilingual engine loaded via ModelManager")
            
        except ImportError as e:
            logger.error(f"❌ Chatterbox not available: {e}")
            raise HTTPException(status_code=503, detail="Chatterbox TTS not installed. pip install chatterbox-tts")
        except Exception as e:
            logger.error(f"❌ Failed to load Chatterbox: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to load Chatterbox: {str(e)}")
    
    return _engine_instances["chatterbox-multilingual"]


def _get_saudi_tts_engine():
    """Get or create Saudi XTTS engine."""
    if "saudi-tts" not in _engine_instances:
        try:
            from ...inference_engines.voice.tts import SaudiXTTSEngine
            
            engine = SaudiXTTSEngine(
                model_path=Path("/home/lumi/.cache/beautyai-models/saudi-tts"),
                speaker_wav_path=Path("/home/lumi/beautyai/backend/speakers/saudi-female/reference.wav")
            )
            
            logger.info("🔄 Loading Saudi XTTS model...")
            if not engine.load_model():
                raise RuntimeError("Failed to load Saudi XTTS model")
            
            _engine_instances["saudi-tts"] = engine
            logger.info("✅ Saudi XTTS engine loaded for testing")
            
        except ImportError as e:
            logger.error(f"❌ Saudi XTTS not available: {e}")
            raise HTTPException(status_code=503, detail="Saudi XTTS not available")
        except Exception as e:
            logger.error(f"❌ Failed to load Saudi XTTS: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to load Saudi XTTS: {str(e)}")
    
    return _engine_instances["saudi-tts"]


@router.get("/engines")
async def list_engines():
    """List available TTS engines and their status."""
    engines = [
        {
            "id": "edge-tts",
            "name": "Microsoft Edge TTS",
            "type": "cloud",
            "languages": ["ar", "en", "fr", "de", "es", "zh", "ja", "ko"],
            "gpu_required": False,
            "available": True,
            "description": "Cloud-based TTS, no GPU required, fast and reliable"
        },
        {
            "id": "saudi-tts",
            "name": "Saudi Arabic XTTS v2",
            "type": "local",
            "languages": ["ar"],
            "gpu_required": True,
            "available": os.path.exists("/home/lumi/.cache/beautyai-models/saudi-tts"),
            "description": "GPU-accelerated, Saudi Arabic dialect, voice cloning"
        },
        {
            "id": "chatterbox-multilingual",
            "name": "Chatterbox Multilingual",
            "type": "local",
            "languages": ["ar", "en", "fr", "de", "es", "zh", "ja", "ko", "ru", "pt", "it", "nl", "pl", "tr", "hi", "he", "da", "fi", "no", "sv", "sw", "ms", "el"],
            "gpu_required": True,
            "available": True,  # Will be checked on first use
            "description": "23 languages, zero-shot voice cloning, state-of-the-art quality"
        }
    ]
    
    return {"engines": engines}


@router.post("/generate", response_model=TTSGenerateResponse)
async def generate_tts(request: TTSGenerateRequest):
    """
    Generate speech from text using the specified TTS engine.
    
    Returns audio as URL to temporary file or base64 encoded.
    """
    start_time = time.time()
    
    try:
        # Validate text
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")
        
        logger.info(f"🎤 TTS request: engine={request.engine}, lang={request.language}, text={text[:50]}...")
        
        # Get appropriate engine
        if request.engine == "edge-tts":
            engine = _get_edge_tts_engine()
        elif request.engine == "chatterbox-multilingual":
            engine = _get_chatterbox_engine()
        elif request.engine == "saudi-tts":
            engine = _get_saudi_tts_engine()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {request.engine}")
        
        # Generate unique output filename
        text_hash = abs(hash(text + request.language + str(time.time()))) % 1000000
        output_path = _tts_output_dir / f"tts_{request.engine}_{request.language}_{text_hash}.wav"
        
        # Generate audio
        if request.engine == "chatterbox-multilingual":
            # Chatterbox with extra parameters
            audio_path = engine.text_to_speech(
                text=text,
                language=request.language,
                output_path=str(output_path),
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                speaker_wav=request.speaker_wav
            )
        else:
            # Standard generation
            audio_path = engine.text_to_speech(
                text=text,
                language=request.language,
                output_path=str(output_path)
            )
        
        generation_time = (time.time() - start_time) * 1000  # ms
        
        # Get audio duration
        duration_seconds = None
        sample_rate = getattr(engine, 'output_sample_rate', 24000)
        
        try:
            import wave
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration_seconds = frames / float(rate)
                sample_rate = rate
        except:
            pass
        
        # Return audio as base64 for simplicity
        with open(audio_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        duration_str = f"{duration_seconds:.2f}s" if duration_seconds is not None else "N/A"
        logger.info(f"✅ TTS generated: {request.engine}, {generation_time:.0f}ms, {duration_str}")
        
        return TTSGenerateResponse(
            success=True,
            engine=request.engine,
            language=request.language,
            audio_base64=audio_base64,
            duration_seconds=duration_seconds,
            generation_time_ms=generation_time,
            sample_rate=sample_rate
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ TTS generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return TTSGenerateResponse(
            success=False,
            engine=request.engine,
            language=request.language,
            error=str(e)
        )
