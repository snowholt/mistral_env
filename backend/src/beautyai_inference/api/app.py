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

# Import WhatsApp Manager routers
try:
    from .endpoints.whatsapp_auth import whatsapp_auth_router
    from .endpoints.whatsapp_manager import whatsapp_manager_router
    from .endpoints.whatsapp_webhook import whatsapp_webhook_router
    from .endpoints.whatsapp_inbox_ws import whatsapp_inbox_ws_router
    whatsapp_routers_available = True
except ImportError as e:
    whatsapp_routers_available = False
    logger.warning(f"WhatsApp Manager routers not available: {e}")



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
    },
    {
        "name": "whatsapp-auth",
        "description": "🔐 **WhatsApp Auth** - User authentication for WhatsApp Manager SaaS platform. "
                      "JWT-based registration, login, and token management."
    },
    {
        "name": "whatsapp-manager",
        "description": "📱 **WhatsApp Manager** - Meta Embedded Signup, AI agent configuration, and inbox management. "
                      "Protected endpoints for business owners to configure their WhatsApp automation."
    },
    {
        "name": "whatsapp-webhook",
        "description": "🔔 **WhatsApp Webhook** - Public endpoint for receiving incoming WhatsApp messages from Meta. "
                      "Integrates with LLM for AI-powered auto-replies."
    },
    {
        "name": "whatsapp-inbox-ws",
        "description": "💬 **WhatsApp Inbox WebSocket** - Real-time inbox updates via WebSocket. "
                      "Authenticated connections receive live message notifications."
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

# Include WhatsApp Manager routers if available
if whatsapp_routers_available:
    app.include_router(whatsapp_auth_router, tags=["whatsapp-auth"])
    app.include_router(whatsapp_manager_router, tags=["whatsapp-manager"])
    app.include_router(whatsapp_webhook_router, tags=["whatsapp-webhook"])
    app.include_router(whatsapp_inbox_ws_router, tags=["whatsapp-inbox-ws"])
    logger.info("✅ WhatsApp Manager endpoints registered:")
    logger.info("   - Auth: /api/v1/whatsapp/auth/*")
    logger.info("   - Manager: /api/v1/whatsapp/*")
    logger.info("   - Webhook: /api/v1/whatsapp/webhook")
    logger.info("   - Inbox WS: /api/v1/whatsapp/inbox/ws")
else:
    logger.warning("WhatsApp Manager endpoints not registered - modules not available")

# Include Admin router
try:
    from .endpoints.admin import admin_router
    app.include_router(admin_router, tags=["admin"])
    logger.info("✅ Admin endpoints registered at /api/v1/admin/*")
except ImportError as e:
    logger.warning(f"Admin router not available: {e}")

# Include Web Chat Widget router
try:
    from .endpoints.webchat import webchat_router
    app.include_router(webchat_router, tags=["webchat"])
    logger.info("✅ Web Chat Widget endpoints registered at /api/v1/webchat/*")
except ImportError as e:
    logger.warning(f"Web Chat Widget router not available: {e}")

# Include Billing router
try:
    from .endpoints.billing import router as billing_router
    app.include_router(billing_router, tags=["billing"])
    logger.info("✅ Billing endpoints registered at /api/v1/billing/*")
except ImportError as e:
    logger.warning(f"Billing router not available: {e}")

# Include Knowledge Base router
try:
    from .endpoints.knowledge_base import router as kb_router
    app.include_router(kb_router, tags=["knowledge-base"])
    logger.info("✅ Knowledge Base endpoints registered at /api/v1/kb/*")
except ImportError as e:
    logger.warning(f"Knowledge Base router not available: {e}")

# Include Prometheus Metrics router
try:
    from .endpoints.metrics import router as metrics_router
    app.include_router(metrics_router, tags=["metrics"])
    logger.info("✅ Prometheus Metrics endpoints registered at /metrics")
except ImportError as e:
    logger.warning(f"Metrics router not available: {e}")

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


@app.get("/webrtc_debug.html", response_class=HTMLResponse)
async def serve_webrtc_debug_page():
    """Serve the WebRTC debug page for connection and audio testing."""
    try:
        # Path to the static file we just copied
        backend_root = Path(__file__).resolve().parents[4]
        test_page_path = backend_root / "backend" / "src" / "beautyai_inference" / "api" / "static" / "webrtc_debug.html"
        
        if not test_page_path.exists():
            raise HTTPException(status_code=404, detail=f"WebRTC debug page not found at {test_page_path}")
        
        with open(test_page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="WebRTC debug page not found")
    except Exception as e:
        logger.error(f"Error serving WebRTC debug page: {e}")
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


@app.get("/test_lean.html", response_class=HTMLResponse)
async def serve_lean_test_page():
    """Serve the lean WebRTC test page."""
    try:
        backend_root = Path(__file__).resolve().parents[4]
        test_page_path = backend_root / "backend" / "src" / "beautyai_inference" / "api" / "static" / "test_lean.html"
        
        if not test_page_path.exists():
            raise HTTPException(status_code=404, detail=f"Lean test page not found at {test_page_path}")
        
        with open(test_page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Lean test page not found")
    except Exception as e:
        logger.error(f"Error serving lean test page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.warning("WebRTC voice endpoints not registered - check aiortc installation")
async def preload_voice_models():
    """Pre-load essential models for WebSocket voice services to improve performance."""
    try:
        logger.info("🚀 Starting Genius AI model preloading from preload_config.json...")
        
        # Use PersistentModelManager to preload models from config
        from ..core.persistent_model_manager import get_persistent_model_manager
        persistent_mgr = get_persistent_model_manager()
        
        # Preload all models defined in preload_config.json
        success = await persistent_mgr.preload_models()
        
        if success:
            logger.info("✅ All Genius AI models pre-loaded successfully from config")
            logger.info("🎯 Models ready: qwen3-unsloth-q4ks, genius-whisper-arabic, genius-xtts-arabic")
        else:
            logger.warning("⚠️ Some models failed to preload - check logs for details")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Critical error during model pre-loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("🚀 BeautyAI Inference API starting up...")
    logger.info("📚 API Documentation available at: http://localhost:8000/docs")
    logger.info("🔍 Alternative docs at: http://localhost:8000/redoc")
    logger.info("🎤 Voice endpoints info at: http://localhost:8000/api/v1/voice/endpoints")
    
    # Initialize WhatsApp Manager database if available
    if whatsapp_routers_available:
        try:
            from ..database.connection import init_db
            await init_db()
            logger.info("📱 WhatsApp Manager database initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize WhatsApp database: {e}")
            logger.info("📱 WhatsApp Manager may have limited functionality")
    
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
    
    # Close WhatsApp Manager database connections
    if whatsapp_routers_available:
        try:
            from ..database.connection import close_db
            await close_db()
            logger.info("📱 WhatsApp Manager database connections closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing WhatsApp database: {e}")
    
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
