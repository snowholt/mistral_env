"""
FastAPI Application for BeautyAI Inference Framework.

This module provides a REST API interface for the BeautyAI inference framework,
including endpoints for model management, chat interactions, and system monitoring.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import logging
import time

# Import the routers
from .endpoints import health_router, models_router, inference_router, config_router, system_router
from .endpoints.websocket_simple_voice import websocket_simple_voice_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define OpenAPI tags for better documentation organization
tags_metadata = [
    {
        "name": "simple-voice",
        "description": "🏎️ **Simple Voice Chat** - Ultra-fast voice conversations with Edge TTS. "
                      "Perfect for real-time chat with <2 second response times. "
                      "Arabic and English support only."
    },
    {
        "name": "health",
        "description": "🏥 **Health & Monitoring** - Service health checks, status monitoring, and system diagnostics."
    },
    {
        "name": "models",
        "description": "🤖 **Model Management** - Load, manage, and monitor AI models and inference engines."
    },
    {
        "name": "inference",
        "description": "💬 **Text Inference** - Text-based chat and completion endpoints for various language models."
    },
    {
        "name": "config",
        "description": "⚙️ **Configuration** - System configuration management and model registry operations."
    },
    {
        "name": "system",
        "description": "🖥️ **System Administration** - System utilities, monitoring, and administrative functions."
    }
]

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="BeautyAI Inference Framework",
    description="""
    🚀 **BeautyAI Voice Services - Optimized for Real-Time Performance**
    
    ## 🏎️ Simple Voice Chat (`/ws/simple-voice-chat`)
    **Ultra-fast voice conversations with Edge TTS**
    - ⚡ **Response Time:** <2 seconds
    - 🌍 **Languages:** Arabic, English
    - 💾 **Memory:** <50MB per connection
    - 🎯 **Use Cases:** Live chat, voice assistants, real-time interactions
    - 🔧 **Models:** Faster-Whisper (large-v3-turbo) + Edge TTS
    
    ### 📊 **Performance Metrics**
    - **STT Speed:** ~1.5 seconds for 10-second audio
    - **TTS Speed:** ~0.5 seconds for short responses
    - **Memory Usage:** <50MB per active connection
    - **GPU Acceleration:** ✅ CUDA-optimized Whisper
    - **Audio Format:** WebM/Opus (optimized for web)
    
    ### 🎯 **Features**
    - **Real-time Voice Chat:** WebSocket-based streaming
    - **Arabic Language Focus:** Optimized for Arabic conversations
    - **Edge TTS Integration:** High-quality voice synthesis
    - **GPU Accelerated:** Faster-Whisper with CUDA support
    - **Minimal Resource Usage:** Designed for efficiency
    
    ---
    
    📚 **API Documentation:** [/docs](/docs) | [/redoc](/redoc)  
    🏥 **Health Checks:** [/health/basic](/health/basic) | [/api/v1/health/voice](/api/v1/health/voice)  
    🎤 **Voice Endpoints Info:** [/api/v1/voice/endpoints](/api/v1/voice/endpoints)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with proper organization
app.include_router(health_router)
app.include_router(models_router)
app.include_router(inference_router)
app.include_router(config_router)
app.include_router(system_router)

# Include voice WebSocket routers with proper prefixes and tags
app.include_router(
    websocket_simple_voice_router,
    prefix="/api/v1",
    tags=["simple-voice"]
)
async def preload_voice_models():
    """Pre-load essential models for WebSocket voice services to improve performance."""
    try:
        # Import model services here to avoid circular imports
        from ..services.model import ModelLifecycleService, RegistryService  
        from ..config.configuration_manager import ConfigurationManager
        from ..config.config_manager import AppConfig
        from pathlib import Path
        
        # Initialize services
        lifecycle_service = ModelLifecycleService()
        registry_service = RegistryService()
        config_manager = ConfigurationManager()
        # Note: Config is already loaded during ConfigurationManager initialization
        
        # Create AppConfig object and point it to the comprehensive model registry
        app_config = AppConfig()
        # Set the correct path to the comprehensive model registry
        app_config.models_file = str(Path(__file__).parent.parent / "config" / "model_registry.json")
        app_config.load_model_registry()  # Load from the comprehensive model registry
       
        # Models to pre-load for voice services
        essential_models = [
            "qwen3-unsloth-q4ks",            # Main chat model
            # Don't pre-load whisper model here - let SimpleVoiceService handle it with base model
        ]
        
        logger.info(f"🔄 Pre-loading {len(essential_models)} essential models...")
        
        for model_name in essential_models:
            try:
                logger.info(f"⏳ Loading {model_name}...")
                
                # Get model config from registry
                model_config = registry_service.get_model(app_config, model_name)
                if not model_config:
                    logger.warning(f"⚠️ Model '{model_name}' not found in registry, skipping")
                    continue
                
                # Check if already loaded
                if lifecycle_service.model_manager.is_model_loaded(model_name):
                    logger.info(f"✅ Model '{model_name}' already loaded")
                    continue
                
                # Load the model
                success, error_msg = lifecycle_service.load_model(model_config, show_progress=False)
                
                if success:
                    logger.info(f"✅ Successfully pre-loaded {model_name}")
                else:
                    logger.warning(f"⚠️ Failed to pre-load {model_name}: {error_msg}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error pre-loading {model_name}: {e}")
                continue
        
        logger.info("🎯 Model pre-loading completed - WebSocket services ready for fast responses")
        
    except Exception as e:
        logger.error(f"❌ Critical error during model pre-loading: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("🚀 BeautyAI Inference API starting up...")
    logger.info("📚 API Documentation available at: http://localhost:8000/docs")
    logger.info("🔍 Alternative docs at: http://localhost:8000/redoc")
    logger.info("🎤 Voice endpoints info at: http://localhost:8000/api/v1/voice/endpoints")
    
    # Pre-load essential models for WebSocket voice services
    logger.info("⏳ Pre-loading essential models for WebSocket voice services...")
    try:
        await preload_voice_models()
        logger.info("✅ Voice models pre-loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to pre-load voice models: {e}")
        logger.warning("⚠️ WebSocket voice services may have slower initial response times")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("🛑 BeautyAI Inference API shutting down...")


@app.get("/")
async def root():
    """Root endpoint with API information and voice endpoint guidance."""
    return {
        "name": "BeautyAI Inference API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "voice_endpoints": {
            "simple_voice_chat": {
                "url": "/api/v1/ws/simple-voice-chat",
                "description": "Ultra-fast voice chat with Edge TTS (<2s response)",
                "best_for": "Real-time conversations, Arabic/English only"
            },
            "advanced_voice_chat": {
                "url": "/api/v1/ws/voice-conversation", 
                "description": "Full-featured voice with Coqui TTS (5-8s response)",
                "best_for": "Voice cloning, 17+ languages, production features"
            },
            "endpoint_comparison": "/api/v1/voice/endpoints",
            "voice_health": "/api/v1/health/voice"
        },
        "other_endpoints": {
            "health": "/health",
            "models": "/models",
            "inference": "/inference",
            "config": "/config", 
            "system": "/system"
        }
    }


@app.get("/api/v1/voice/endpoints", tags=["simple-voice"])
async def get_voice_endpoints():
    """
    🎤 **Voice Endpoint Information**
    
    Get comprehensive information about the voice endpoint and usage recommendations.
    This endpoint provides details about the simple voice service configuration.
    """
    return {
        "endpoints": {
            "simple_voice_chat": {
                "url": "/api/v1/ws/simple-voice-chat",
                "type": "WebSocket",
                "engine": "Edge TTS + Faster-Whisper",
                "performance": {
                    "response_time": "< 2 seconds",
                    "memory_usage": "< 50MB",
                    "setup_time": "< 100ms",
                    "connection_overhead": "Minimal"
                },
                "features": {
                    "languages": ["ar", "en"],
                    "voice_types": ["male", "female"],
                    "real_time": True,
                    "gpu_accelerated": True,
                    "audio_format": "webm/opus",
                    "session_management": "Simplified"
                },
                "parameters": {
                    "required": ["language", "voice_type"],
                    "optional": ["session_id"],
                    "total_count": 3
                },
                "best_for": [
                    "Real-time voice chat",
                    "Quick voice interactions", 
                    "Speed-critical applications",
                    "Resource-constrained environments",
                    "Voice assistants",
                    "Live customer support"
                ],
                "models": {
                    "stt": "openai/whisper-large-v3-turbo",
                    "tts": "Microsoft Edge TTS",
                    "gpu_optimized": True
                }
            }
        },
        "usage_guidelines": {
            "when_to_use": [
                "Response time < 3 seconds required",
                "Arabic/English conversations", 
                "Memory usage < 100MB required",
                "Real-time conversation needed",
                "GPU acceleration available",
                "Fast deployment needed"
            ],
            "features": {
                "response_time": "< 2 seconds",
                "memory_usage": "< 50MB", 
                "languages": ["Arabic", "English"],
                "setup_complexity": "3 parameters",
                "deployment": "Instant"
            }
        },
        "usage_examples": {
            "simple_voice_connection": {
                "url": "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female",
                "description": "Connect for fast Arabic female voice chat",
                "expected_response_time": "< 2 seconds"
            }
        },
        "metadata": {
            "last_updated": time.time(),
            "version": "2.0.0",
            "total_endpoints": 1,
            "documentation_url": "/docs"
        }
    }


@app.get("/api/v1/health/voice", tags=["health"])
async def health_check_voice():
    """
    🏥 **Enhanced Voice Services Health Check**
    
    Comprehensive health check for both voice services with status monitoring,
    performance metrics, and connection information.
    """
    # Check simple voice service status
    simple_status = await check_simple_voice_service()
    
    # Calculate overall health
    overall_healthy = simple_status["healthy"]
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": time.time(),
        "services": {
            "simple_voice": {
                "name": "Simple Voice Chat",
                "endpoint": "/api/v1/ws/simple-voice-chat",
                "status": "available" if simple_status["healthy"] else "unavailable",
                "engine": "Edge TTS via SimpleVoiceService",
                "performance": {
                    "target_response_time": "< 2 seconds",
                    "target_memory_usage": "< 50MB",
                    "supported_languages": ["ar", "en"],
                    "voice_types": ["male", "female"]
                },
                "connections": simple_status.get("connections", {}),
                "ready_for_connections": simple_status["healthy"],
                "last_check": simple_status.get("last_check", time.time())
            }
        },
        "overall_metrics": {
            "total_active_connections": simple_status.get("connections", {}).get("count", 0),
            "services_available": 1 if simple_status["healthy"] else 0,
            "services_total": 1,
            "uptime_percentage": 100 if overall_healthy else 0
        },
        "recommendations": {
            "use_simple_voice": simple_status["healthy"] and "For real-time conversations with Arabic/English",
            "fallback_options": [
                "If service down, check system health at /health/detailed",
                "For voice issues, restart the service or check model availability"
            ]
        }
    }


async def check_simple_voice_service() -> Dict[str, Any]:
    """Check the health of the simple voice service."""
    try:
        # Import here to avoid circular imports
        from .endpoints.websocket_simple_voice import simple_voice_connections, simple_ws_manager
        
        # Check if service can be initialized
        service_healthy = True
        error_message = None
        
        try:
            # Test service initialization (lightweight check)
            await simple_ws_manager._ensure_service_initialized()
        except Exception as e:
            service_healthy = False
            error_message = str(e)
            logger.warning(f"Simple voice service health check failed: {e}")
        
        return {
            "healthy": service_healthy,
            "error": error_message,
            "connections": {
                "count": len(simple_voice_connections),
                "active_sessions": list(simple_voice_connections.keys())
            },
            "last_check": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to check simple voice service health: {e}")
        return {
            "healthy": False,
            "error": f"Health check failed: {str(e)}",
            "connections": {"count": 0, "active_sessions": []},
            "last_check": time.time()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
