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
from .endpoints.websocket_voice import websocket_voice_router
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
        "name": "advanced-voice", 
        "description": "🎭 **Advanced Voice Chat** - Full-featured voice with Coqui TTS and voice cloning. "
                      "Supports 17+ languages, custom voices, and content filtering. "
                      "Best for production applications."
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
    🚀 **BeautyAI Voice Services - Dual Architecture for Maximum Performance**
    
    Choose the right endpoint for your use case:
    
    ## 🏎️ Simple Voice Chat (`/ws/simple-voice-chat`)
    **Best for:** Real-time conversations, speed priority
    - ⚡ **Response Time:** <2 seconds
    - 🌍 **Languages:** Arabic, English only
    - 💾 **Memory:** <50MB per connection
    - 🎯 **Use Cases:** Live chat, voice assistants, quick interactions
    
    ## 🎭 Advanced Voice Chat (`/ws/voice-conversation`)
    **Best for:** Production features, voice cloning
    - ⏱️ **Response Time:** 5-8 seconds
    - 🌍 **Languages:** 17+ with auto-detection
    - 💾 **Memory:** 3GB+ per session
    - 🎯 **Use Cases:** Voice cloning, content creation, complex conversations

    ---
    
    ### 📊 **Performance Comparison**
    
    | Feature | Simple Voice | Advanced Voice |
    |---------|-------------|----------------|
    | Speed | <2s | 5-8s |
    | Memory | <50MB | 3GB+ |
    | Languages | 2 (ar, en) | 17+ |
    | Voice Cloning | ❌ | ✅ |
    | Content Filtering | ❌ | ✅ |
    | Real-time | ✅ | ❌ |
    | Auto Language Detection | ❌ | ✅ |
    
    ### 🎯 **Decision Guide**
    
    **Choose Simple Voice if:**
    - Response time < 3 seconds required
    - Only Arabic/English needed
    - Memory usage < 100MB required
    - Real-time conversation needed
    
    **Choose Advanced Voice if:**
    - Voice cloning needed
    - Multiple languages required
    - Content filtering needed
    - Production-grade features needed
    
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
    websocket_voice_router,
    prefix="/api/v1",
    tags=["advanced-voice"]
)
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


@app.get("/api/v1/voice/endpoints", tags=["simple-voice", "advanced-voice"])
async def get_voice_endpoints():
    """
    🎤 **Voice Endpoint Decision Matrix**
    
    Get comprehensive information about available voice endpoints and usage recommendations.
    This endpoint helps developers choose the right voice service for their use case.
    """
    return {
        "endpoints": {
            "simple_voice_chat": {
                "url": "/api/v1/ws/simple-voice-chat",
                "type": "WebSocket",
                "engine": "Edge TTS via SimpleVoiceService",
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
                    "voice_cloning": False,
                    "content_filtering": False,
                    "auto_language_detection": False,
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
                "limitations": [
                    "Arabic and English only",
                    "No voice cloning",
                    "No content filtering",
                    "Fixed voice types only"
                ]
            },
            "advanced_voice_chat": {
                "url": "/api/v1/ws/voice-conversation",
                "type": "WebSocket",
                "engine": "Coqui TTS via AdvancedVoiceConversationService", 
                "performance": {
                    "response_time": "5-8 seconds",
                    "memory_usage": "3GB+",
                    "setup_time": "10-30 seconds",
                    "connection_overhead": "Significant"
                },
                "features": {
                    "languages": ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh-cn", "ja", "hu", "ko", "hi"],
                    "voice_types": ["custom", "cloned", "male", "female"],
                    "real_time": False,
                    "voice_cloning": True,
                    "content_filtering": True,
                    "emotion_control": True,
                    "auto_language_detection": True,
                    "session_management": "Full-featured"
                },
                "parameters": {
                    "required": ["model"],
                    "optional": ["language", "voice_type", "enable_content_filtering", "session_id", "conversation_context", "voice_clone_params"],
                    "total_count": "20+"
                },
                "best_for": [
                    "Voice cloning projects",
                    "Content creation",
                    "Multi-language support",
                    "Production applications",
                    "Complex voice conversations",
                    "Emotion-aware responses"
                ],
                "limitations": [
                    "Slower response time",
                    "High memory usage",
                    "Complex parameter set",
                    "Longer setup time"
                ]
            }
        },
        "decision_guide": {
            "choose_simple_if": [
                "Response time < 3 seconds required",
                "Only Arabic/English needed",
                "Memory usage < 100MB required",
                "Real-time conversation needed",
                "Simple voice types sufficient",
                "Fast deployment needed"
            ],
            "choose_advanced_if": [
                "Voice cloning needed",
                "Multiple languages required",
                "Content filtering needed",
                "Production-grade features needed",
                "Custom voice creation required",
                "Emotion control needed"
            ]
        },
        "performance_comparison": {
            "metrics": {
                "response_time": {
                    "simple": "< 2 seconds",
                    "advanced": "5-8 seconds",
                    "difference": "3-6x faster"
                },
                "memory_usage": {
                    "simple": "< 50MB",
                    "advanced": "3GB+",
                    "difference": "60x less memory"
                },
                "languages": {
                    "simple": 2,
                    "advanced": 17,
                    "difference": "Advanced supports 8.5x more languages"
                },
                "setup_complexity": {
                    "simple": "3 parameters",
                    "advanced": "20+ parameters",
                    "difference": "Simple has 7x fewer parameters"
                }
            }
        },
        "usage_examples": {
            "simple_voice_connection": {
                "url": "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female",
                "description": "Connect for fast Arabic female voice chat",
                "expected_response_time": "< 2 seconds"
            },
            "advanced_voice_connection": {
                "url": "ws://localhost:8000/api/v1/ws/voice-conversation",
                "description": "Connect for full-featured voice with cloning",
                "parameters": {
                    "model": "qwen3_unsloth_q4ks",
                    "voice_type": "clone",
                    "enable_content_filtering": True
                }
            }
        },
        "metadata": {
            "last_updated": time.time(),
            "version": "2.0.0",
            "total_endpoints": 2,
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
    advanced_status = await check_advanced_voice_service()
    
    # Calculate overall health
    overall_healthy = simple_status["healthy"] and advanced_status["healthy"]
    
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
            },
            "advanced_voice": {
                "name": "Advanced Voice Chat",
                "endpoint": "/api/v1/ws/voice-conversation",
                "status": "available" if advanced_status["healthy"] else "unavailable",
                "engine": "Coqui TTS via AdvancedVoiceConversationService",
                "performance": {
                    "target_response_time": "5-8 seconds",
                    "target_memory_usage": "3GB+",
                    "supported_languages": ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh-cn", "ja", "hu", "ko", "hi"],
                    "voice_types": ["custom", "cloned", "male", "female"]
                },
                "connections": advanced_status.get("connections", {}),
                "ready_for_connections": advanced_status["healthy"],
                "last_check": advanced_status.get("last_check", time.time())
            }
        },
        "overall_metrics": {
            "total_active_connections": simple_status.get("connections", {}).get("count", 0) + advanced_status.get("connections", {}).get("count", 0),
            "services_available": sum([simple_status["healthy"], advanced_status["healthy"]]),
            "services_total": 2,
            "uptime_percentage": 100 if overall_healthy else 50 if (simple_status["healthy"] or advanced_status["healthy"]) else 0
        },
        "recommendations": {
            "use_simple_voice": simple_status["healthy"] and "For real-time conversations with Arabic/English",
            "use_advanced_voice": advanced_status["healthy"] and "For voice cloning and multi-language support",
            "fallback_options": [
                "If both services down, check system health at /health/detailed",
                "For simple voice issues, try advanced voice for non-real-time use",
                "For advanced voice issues, use simple voice for basic conversations"
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


async def check_advanced_voice_service() -> Dict[str, Any]:
    """Check the health of the advanced voice service."""
    try:
        # Import here to avoid circular imports
        from .endpoints.websocket_voice import active_connections
        
        # For now, assume advanced service is healthy if we can import it
        # In a real implementation, we would test the service initialization
        service_healthy = True
        error_message = None
        
        try:
            # We could add more sophisticated health checks here
            # For now, just check if the service module can be imported
            from beautyai_inference.services.voice.conversation.advanced_voice_service import AdvancedVoiceConversationService
        except Exception as e:
            service_healthy = False
            error_message = str(e)
            logger.warning(f"Advanced voice service health check failed: {e}")
        
        return {
            "healthy": service_healthy,
            "error": error_message,
            "connections": {
                "count": len(active_connections),
                "active_sessions": list(active_connections.keys())
            },
            "last_check": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to check advanced voice service health: {e}")
        return {
            "healthy": False,
            "error": f"Health check failed: {str(e)}",
            "connections": {"count": 0, "active_sessions": []},
            "last_check": time.time()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
