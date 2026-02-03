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
