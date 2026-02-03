"""
PersonaPlex Voice API Endpoints

REST API endpoints for controlling the PersonaPlex full-duplex
speech-to-speech server.

Endpoints:
- POST /api/v1/personaplex/start - Start PersonaPlex server
- POST /api/v1/personaplex/stop - Stop PersonaPlex server  
- GET /api/v1/personaplex/status - Get server status
- GET /api/v1/personaplex/voices - List available voice prompts
- GET /api/v1/personaplex/prompts - List available text prompts
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...services.personaplex import (
    PersonaPlexManager,
    get_personaplex_manager,
    PersonaPlexConfig,
    VoiceType,
    VOICE_PROMPTS,
    DEFAULT_TEXT_PROMPTS,
)

logger = logging.getLogger(__name__)

personaplex_router = APIRouter(
    prefix="/api/v1/personaplex",
    tags=["personaplex", "voice", "s2s"]
)


# ============================================
# Request/Response Models
# ============================================

class StartServerRequest(BaseModel):
    """Request to start PersonaPlex server."""
    voice_prompt: Optional[str] = Field(
        default="NATF2",
        description="Voice prompt to use (e.g., NATF2, NATM1)"
    )
    text_prompt: Optional[str] = Field(
        default="assistant",
        description="Text prompt key (e.g., assistant, casual) or custom prompt text"
    )
    cpu_offload: bool = Field(
        default=True,
        description="Enable CPU offload for limited VRAM (required <14GB)"
    )


class StartServerResponse(BaseModel):
    """Response after starting PersonaPlex server."""
    success: bool
    message: str
    url: Optional[str] = None
    webui_url: Optional[str] = None
    status: str
    pid: Optional[int] = None
    cpu_offload: Optional[bool] = None


class StopServerResponse(BaseModel):
    """Response after stopping PersonaPlex server."""
    success: bool
    message: str
    status: str


class ServerStatusResponse(BaseModel):
    """Detailed server status response."""
    status: str
    is_running: bool
    url: Optional[str] = None
    webui_url: Optional[str] = None
    pid: Optional[int] = None
    uptime_seconds: Optional[float] = None
    error: Optional[str] = None
    active_sessions: int = 0
    config: dict


class VoicesResponse(BaseModel):
    """Available voice prompts response."""
    voices: dict
    default: str
    categories: dict
    recommended: dict


class TextPromptsResponse(BaseModel):
    """Available text prompts response."""
    prompts: dict
    default: str
    categories: dict


# ============================================
# Endpoints
# ============================================

@personaplex_router.post("/start", response_model=StartServerResponse)
async def start_personaplex_server(request: StartServerRequest = StartServerRequest()):
    """
    Start the PersonaPlex full-duplex speech-to-speech server.
    
    PersonaPlex is NVIDIA's real-time conversational AI model that enables
    natural voice conversations with interruption support, backchanneling,
    and smooth turn-taking.
    
    **Prerequisites:**
    - PersonaPlex cloned to /home/lumi/personaplex
    - HF_TOKEN environment variable set
    - Existing models unloaded (call /api/v1/admin/unload-all-models first)
    
    **VRAM Requirements:**
    - With cpu_offload=true: ~7-10GB VRAM + 10-14GB RAM
    - With cpu_offload=false: ~14GB VRAM
    
    After starting, access the WebUI at the returned `webui_url`.
    """
    try:
        manager = get_personaplex_manager()
        
        # Update config based on request
        manager.config.cpu_offload = request.cpu_offload
        
        # Start server
        result = await manager.start_server(
            voice_prompt=VoiceType(request.voice_prompt) if request.voice_prompt in [v.value for v in VoiceType] else None,
            text_prompt=request.text_prompt,
        )
        
        return StartServerResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to start PersonaPlex: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@personaplex_router.post("/stop", response_model=StopServerResponse)
async def stop_personaplex_server():
    """
    Stop the PersonaPlex server.
    
    This will:
    - Gracefully terminate the server process
    - Clean up SSL certificates
    - Free GPU/CPU memory
    """
    try:
        manager = get_personaplex_manager()
        result = await manager.stop_server()
        return StopServerResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to stop PersonaPlex: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@personaplex_router.get("/status", response_model=ServerStatusResponse)
async def get_personaplex_status():
    """
    Get PersonaPlex server status.
    
    Returns:
    - Current status (stopped, starting, running, error)
    - Server URLs (if running)
    - Process info
    - Configuration
    """
    try:
        manager = get_personaplex_manager()
        result = manager.get_status()
        return ServerStatusResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get PersonaPlex status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@personaplex_router.get("/voices", response_model=VoicesResponse)
async def get_available_voices():
    """
    Get available voice prompts for PersonaPlex.
    
    Voice categories:
    - **Natural**: More conversational, recommended for most use cases
      - Female: NATF0-3 (NATF2 recommended)
      - Male: NATM0-3 (NATM1 recommended)
    - **Variety**: More expressive and unique voices
      - Female: VARF0-4
      - Male: VARM0-4
    """
    try:
        manager = get_personaplex_manager()
        result = manager.get_voices()
        return VoicesResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@personaplex_router.get("/prompts", response_model=TextPromptsResponse)
async def get_available_prompts():
    """
    Get available text prompts for PersonaPlex personas.
    
    Prompt categories:
    - **General**: assistant, casual
    - **Customer Service**: bank, medical, restaurant, rental
    - **Roleplay**: astronaut, beauty_consultant
    
    You can also provide custom text prompts when starting the server.
    """
    try:
        manager = get_personaplex_manager()
        result = manager.get_text_prompts()
        return TextPromptsResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@personaplex_router.get("/health")
async def health_check():
    """
    Quick health check for PersonaPlex server.
    
    Returns simple status for monitoring/load balancers.
    """
    manager = get_personaplex_manager()
    return {
        "healthy": manager.is_running,
        "status": manager.status.value,
    }


@personaplex_router.get("/info")
async def get_personaplex_info():
    """
    Get information about PersonaPlex model and capabilities.
    """
    return {
        "name": "PersonaPlex",
        "version": "7B v1",
        "description": "NVIDIA PersonaPlex - Voice and Role Control for Full Duplex Conversational Speech Models",
        "model_id": "nvidia/personaplex-7b-v1",
        "parameters": "7B",
        "capabilities": [
            "Full-duplex speech-to-speech conversation",
            "Voice cloning via audio prompts",
            "Role control via text prompts",
            "Natural interruption handling",
            "Backchanneling ('uh-huh', 'okay', etc.)",
            "Smooth turn-taking",
        ],
        "supported_languages": ["en"],
        "documentation": {
            "github": "https://github.com/NVIDIA/personaplex",
            "paper": "https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf",
            "huggingface": "https://huggingface.co/nvidia/personaplex-7b-v1",
        },
        "hardware_requirements": {
            "min_vram_gb": 9,
            "recommended_vram_gb": 14,
            "cpu_offload_available": True,
            "cpu_offload_ram_gb": 14,
        },
        "license": "NVIDIA Open Model License",
    }


# ============================================
# Model Management Endpoints (No Auth Required for Dev)
# ============================================

@personaplex_router.get("/models")
async def get_loaded_models():
    """
    Get list of currently loaded models with memory info.
    
    This endpoint is useful for checking which models are consuming
    VRAM before starting PersonaPlex.
    """
    from ...core.persistent_model_manager import get_persistent_model_manager
    from ...utils.memory_utils import get_gpu_memory_stats
    import gc
    
    manager = get_persistent_model_manager()
    
    # Get model info
    models = manager.get_loaded_models_info()
    
    # Get GPU memory stats
    gpu_stats = get_gpu_memory_stats()
    gpu_info = gpu_stats[0] if gpu_stats else {}
    
    return {
        "models": models,
        "total_models": len(models),
        "gpu_memory": {
            "used_mb": gpu_info.get("memory_used_mb", 0),
            "free_mb": gpu_info.get("memory_free_mb", 0),
            "total_mb": gpu_info.get("memory_total_mb", 0),
            "utilization_percent": gpu_info.get("gpu_utilization", 0),
        },
        "personaplex_requirements": {
            "min_free_vram_mb": 9000,
            "recommended_free_vram_mb": 14000,
        }
    }


@personaplex_router.post("/models/unload/{model_id}")
async def unload_model(model_id: str):
    """
    Unload a specific model to free GPU VRAM.
    
    Model IDs:
    - `stt` or `whisper` - Speech-to-Text model (~3GB VRAM)
    - `llm` - All LLM instances (~8GB VRAM each)
    - `llm:0`, `llm:1` - Specific LLM instance
    - `tts` - Text-to-Speech model (varies)
    
    Use this to free VRAM before starting PersonaPlex.
    """
    from ...core.persistent_model_manager import get_persistent_model_manager
    from ...utils.memory_utils import get_gpu_memory_stats, clear_gpu_memory
    import gc
    
    manager = get_persistent_model_manager()
    
    # Get memory before
    gpu_before = get_gpu_memory_stats()
    memory_before = gpu_before[0] if gpu_before else {}
    
    # Check if model exists
    if not manager.is_model_loaded(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' is not currently loaded"
        )
    
    logger.info(f"🧹 Unloading model: {model_id}")
    
    # Unload the model
    try:
        success = await manager.unload_model(model_id)
    except Exception as e:
        logger.error(f"Failed to unload model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )
    
    # Force cleanup
    gc.collect()
    clear_gpu_memory()
    
    # Get memory after
    gpu_after = get_gpu_memory_stats()
    memory_after = gpu_after[0] if gpu_after else {}
    
    freed_mb = max(0, memory_before.get('memory_used_mb', 0) - 
                memory_after.get('memory_used_mb', 0))
    
    logger.info(f"✅ Model {model_id} unloaded. Freed ~{freed_mb:.0f}MB VRAM")
    
    return {
        "success": success,
        "message": f"Model '{model_id}' unloaded. Freed ~{freed_mb:.0f}MB GPU memory.",
        "model_id": model_id,
        "memory_before": {
            "used_mb": memory_before.get('memory_used_mb', 0),
            "free_mb": memory_before.get('memory_free_mb', 0),
            "total_mb": memory_before.get('memory_total_mb', 0),
        },
        "memory_after": {
            "used_mb": memory_after.get('memory_used_mb', 0),
            "free_mb": memory_after.get('memory_free_mb', 0),
            "total_mb": memory_after.get('memory_total_mb', 0),
        },
        "freed_mb": freed_mb,
    }


@personaplex_router.post("/models/unload-all")
async def unload_all_models():
    """
    Unload all models to free GPU VRAM for PersonaPlex.
    
    This is the recommended way to prepare for PersonaPlex:
    1. Call this endpoint to free VRAM
    2. Wait a moment for cleanup
    3. Start PersonaPlex server
    
    Returns memory stats before and after cleanup.
    """
    from ...core.persistent_model_manager import get_persistent_model_manager, cleanup_persistent_models
    from ...utils.memory_utils import get_gpu_memory_stats, clear_gpu_memory
    import gc
    
    # Get memory before cleanup
    gpu_before = get_gpu_memory_stats()
    memory_before = gpu_before[0] if gpu_before else {}
    
    # Get manager and check what's loaded
    manager = get_persistent_model_manager()
    models_status_before = manager.check_models_ready()
    
    logger.info(f"🧹 Unloading all models for PersonaPlex...")
    logger.info(f"   Models before: {list(manager._preloaded_models.keys())}")
    
    # Perform cleanup
    success = await cleanup_persistent_models()
    
    # Force additional cleanup
    gc.collect()
    clear_gpu_memory()
    
    # Small delay for GPU memory to be released
    import asyncio
    await asyncio.sleep(0.5)
    
    # Get memory after cleanup
    gpu_after = get_gpu_memory_stats()
    memory_after = gpu_after[0] if gpu_after else {}
    
    # Calculate freed memory
    freed_mb = max(0, memory_before.get('memory_used_mb', 0) - 
                memory_after.get('memory_used_mb', 0))
    
    logger.info(f"✅ All models unloaded. Freed ~{freed_mb:.0f}MB VRAM")
    
    return {
        "success": success,
        "message": f"All models unloaded. Freed ~{freed_mb:.0f}MB GPU memory.",
        "models_unloaded": list(models_status_before.keys()),
        "memory_before": {
            "used_mb": memory_before.get('memory_used_mb', 0),
            "free_mb": memory_before.get('memory_free_mb', 0),
            "total_mb": memory_before.get('memory_total_mb', 0),
        },
        "memory_after": {
            "used_mb": memory_after.get('memory_used_mb', 0),
            "free_mb": memory_after.get('memory_free_mb', 0),
            "total_mb": memory_after.get('memory_total_mb', 0),
        },
        "freed_mb": freed_mb,
        "ready_for_personaplex": memory_after.get('memory_free_mb', 0) >= 9000,
    }
