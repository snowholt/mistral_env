"""
FastAPI Application for BeautyAI Inference Framework.

This module provides a REST API interface for the BeautyAI inference framework,
including endpoints for model management, chat interactions, and system monitoring.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict, Any
import logging
import time
import os
from pathlib import Path

# Logging configured centrally in run_server via configure_logging.
logger = logging.getLogger(__name__)

# Import the routers
from .endpoints import health_router, models_router, inference_router, config_router, system_router, streaming_voice_router
from .endpoints.debug_router import debug_router
from .endpoints.websocket_simple_voice import websocket_simple_voice_router
from .endpoints.ws_audio_debug import ws_audio_debug_router

# Import WebRTC voice router (Phase B - WebRTC Migration)
try:
    from .endpoints.webrtc_voice import webrtc_voice_router
    webrtc_router_available = True
except ImportError as e:
    webrtc_router_available = False
    logger.warning(f"WebRTC voice router not available - WebRTC features disabled: {e}")

# Import WebRTC debug capture router
try:
    from .endpoints.webrtc_debug_capture import debug_capture_router
    debug_capture_router_available = True
except ImportError as e:
    debug_capture_router_available = False
    logger.warning(f"WebRTC debug capture router not available: {e}")



# Import performance dashboard router
try:
    from .endpoints.performance_dashboard import performance_router
    performance_router_available = True
except ImportError as e:
    performance_router_available = False
    logger.warning(f"Performance dashboard router not available: {e}")

# Define OpenAPI tags for better documentation organization
tags_metadata = [
    {
        "name": "simple-voice",
        "description": "🏎️ **Simple Voice Chat** - Ultra-fast voice conversations with Edge TTS. "
                      "Perfect for real-time chat with <2 second response times. "
                      "Arabic and English support only."
    },
    {
        "name": "webrtc-voice",
        "description": "🌐 **WebRTC Voice** - Browser-based WebRTC voice-to-voice signaling endpoints. "
                      "Supports SDP offer/answer exchange, ICE candidates, and peer connection management. "
                      "Enables high-quality, low-latency voice communication with built-in audio processing."
    },
    {
        "name": "webrtc-debug",
        "description": "🐛 **WebRTC Debug** - Audio capture debugging tools for analyzing sample rates, resampling, "
                      "and audio pipeline issues. Saves audio at each processing layer without STT/LLM overhead."
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
    },
    {
        "name": "performance",
        "description": "📊 **Performance Monitoring** - Real-time performance metrics, alerts, and system analytics."
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
    - **Thinking Mode Support:** Use /think or /no_think in messages
    
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

from .middleware.correlation import CorrelationIdMiddleware, WebSocketCorrelationMiddleware

# Add CORS middleware for WebRTC and cross-origin requests
default_cors_origins = [
    "https://web.lumidev.ca",
    "https://api.lumidev.ca",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = default_cors_origins

# Deduplicate while preserving order
seen_origins = set()
filtered_origins = []
for origin in allowed_origins:
    if origin not in seen_origins:
        filtered_origins.append(origin)
        seen_origins.add(origin)

proxy_handles_cors = os.getenv("PROXY_HANDLES_CORS", "0") == "1"
if proxy_handles_cors:
    logger.info("Skipping FastAPI CORS middleware (proxy handles CORS headers)")
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=filtered_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

# Correlation / request ID injection
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(WebSocketCorrelationMiddleware)

# Include routers with proper organization
app.include_router(health_router)
app.include_router(models_router)
app.include_router(inference_router)
app.include_router(config_router)
app.include_router(system_router)
app.include_router(debug_router)

# Include performance dashboard router if available
if performance_router_available:
    app.include_router(
        performance_router,
        tags=["performance"]
    )
    logger.info("Performance dashboard endpoints registered")
else:
    logger.warning("Performance dashboard endpoints not registered - module not available")

# Conditionally include streaming voice scaffold (Phase 1) if feature flag set and router imported.
if streaming_voice_router is not None:  # pragma: no cover (env dependent)
    app.include_router(
        streaming_voice_router,
        prefix="/api/v1",
        tags=["streaming-voice"],
    )

# Include voice WebSocket routers with proper prefixes and tags
app.include_router(
    websocket_simple_voice_router,
    prefix="/api/v1",
    tags=["simple-voice"]
)

# Include WebSocket audio debug router
app.include_router(
    ws_audio_debug_router,
    tags=["debug"]
)

# Include WebRTC voice router if available (Phase B - WebRTC Migration)
if webrtc_router_available:
    app.include_router(
        webrtc_voice_router,
        tags=["webrtc-voice"]
    )
    logger.info("WebRTC voice endpoints registered at /api/v1/webrtc/voice")
else:
    logger.warning("WebRTC voice endpoints not registered - module not available")

# Include WebRTC debug capture router if available
if debug_capture_router_available:
    app.include_router(
        debug_capture_router,
        tags=["webrtc-debug"]
    )
    logger.info("WebRTC debug capture endpoints registered at /api/v1/webrtc/debug/voice-capture")
else:
    logger.warning("WebRTC debug capture endpoints not registered - module not available")

# Serve debug test page
@app.get("/webrtc_voice_capture_test.html", response_class=HTMLResponse)
async def serve_debug_test_page():
    """Serve the WebRTC debug audio capture test page."""
    try:
        # Path relative to backend/src/beautyai_inference/api/app.py
        # Go up 4 levels to reach project root: api -> beautyai_inference -> src -> backend -> root
        backend_root = Path(__file__).resolve().parents[4]
        test_page_path = backend_root / "frontend" / "src" / "templates" / "webrtc_voice_capture_test.html"
        
        if not test_page_path.exists():
            raise HTTPException(status_code=404, detail=f"Test page not found at {test_page_path}")
        
        with open(test_page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="WebRTC debug test page not found")
    except Exception as e:
        logger.error(f"Error serving debug test page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test_simple.html", response_class=HTMLResponse)
async def serve_simple_test_page():
    """Serve the simplified WebRTC test page for connection debugging."""
    try:
        # Path to the static file we just copied
        backend_root = Path(__file__).resolve().parents[4]
        test_page_path = backend_root / "backend" / "src" / "beautyai_inference" / "api" / "static" / "test_simple.html"
        
        if not test_page_path.exists():
            raise HTTPException(status_code=404, detail=f"Simple test page not found at {test_page_path}")
        
        with open(test_page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Simple WebRTC test page not found")
    except Exception as e:
        logger.error(f"Error serving simple test page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test_ws.html", response_class=HTMLResponse)
async def serve_websocket_test_page():
    """Serve the WebSocket audio test page (no WebRTC)."""
    try:
        backend_root = Path(__file__).resolve().parents[4]
        test_page_path = backend_root / "backend" / "src" / "beautyai_inference" / "api" / "static" / "test_ws.html"
        
        if not test_page_path.exists():
            raise HTTPException(status_code=404, detail=f"WebSocket test page not found at {test_page_path}")
        
        with open(test_page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="WebSocket test page not found")
    except Exception as e:
        logger.error(f"Error serving WebSocket test page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test_lean_capture.html", response_class=HTMLResponse)
async def serve_lean_capture_test_page():
    """Serve the lean capture test page with hardened architecture."""
    try:
        backend_root = Path(__file__).resolve().parents[4]
        test_page_path = backend_root / "backend" / "src" / "beautyai_inference" / "api" / "static" / "test_lean_capture.html"
        
        if not test_page_path.exists():
            raise HTTPException(status_code=404, detail=f"Lean capture test page not found at {test_page_path}")
        
        with open(test_page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Lean capture test page not found")
    except Exception as e:
        logger.error(f"Error serving lean capture test page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.warning("WebRTC voice endpoints not registered - check aiortc installation")
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
    
    # Initialize WebRTC connection pool if available (Phase B - WebRTC Migration)
    if webrtc_router_available:
        try:
            from ..core.webrtc_connection_pool import initialize_webrtc_pool
            await initialize_webrtc_pool()
            logger.info("🌐 WebRTC connection pool initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize WebRTC connection pool: {e}")
            logger.info("🌐 Continuing without WebRTC support")
    
    # Initialize performance monitoring system
    try:
        from ..api.performance_integration import initialize_performance_monitoring
        success = await initialize_performance_monitoring()
        if success:
            logger.info("📊 Performance monitoring system initialized successfully")
        else:
            logger.warning("📊 Performance monitoring initialization failed")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize performance monitoring: {e}")
        logger.info("📊 Continuing without performance monitoring")
    
    # Initialize buffer optimization system
    try:
        from ..core.buffer_integration import initialize_buffer_optimization_from_config
        buffer_manager = await initialize_buffer_optimization_from_config()
        if buffer_manager:
            logger.info("📊 Buffer optimization system initialized successfully")
        else:
            logger.info("📊 Buffer optimization disabled in configuration")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize buffer optimization: {e}")
        logger.info("📊 Continuing without buffer optimization")
    
    # Check if model preloading should be skipped (useful for development/testing)
    skip_preload = os.getenv("SKIP_MODEL_PRELOAD", "0") == "1"
    if skip_preload:
        logger.info("⏭️ Skipping model pre-loading (SKIP_MODEL_PRELOAD=1)")
        return
    
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
    
    # Shutdown WebRTC connection pool if available (Phase B - WebRTC Migration)
    if webrtc_router_available:
        try:
            from ..core.webrtc_connection_pool import shutdown_webrtc_pool
            await shutdown_webrtc_pool()
            logger.info("🌐 WebRTC connection pool shut down successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error shutting down WebRTC connection pool: {e}")
    
    # Shutdown performance monitoring system
    try:
        from ..api.performance_integration import shutdown_performance_monitoring
        await shutdown_performance_monitoring()
        logger.info("📊 Performance monitoring system shut down successfully")
    except Exception as e:
        logger.warning(f"⚠️ Error shutting down performance monitoring: {e}")
    
    # Shutdown buffer optimization system
    try:
        from ..core.buffer_optimizer import shutdown_buffer_manager
        await shutdown_buffer_manager()
        logger.info("📊 Buffer optimization system shut down successfully")
    except Exception as e:
        logger.warning(f"⚠️ Error shutting down buffer optimization: {e}")


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
