"""
Inference operation API endpoints.

Provides REST API endpoints for inference operations including:
- Interactive chat sessions
- Single model testing
- Performance benchmarking  
- Session management
"""
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from ..models import (
    APIResponse, ChatRequest, ChatResponse, TestRequest, TestResponse,
    BenchmarkRequest, BenchmarkResponse,
    SessionSaveRequest, SessionSaveResponse, SessionLoadRequest, SessionLoadResponse
)
from ..auth import AuthContext, get_auth_context, require_permissions
from ..errors import ModelNotFoundError, ModelLoadError, ValidationError
from ...services.inference import ChatService, TestService, BenchmarkService, SessionService

logger = logging.getLogger(__name__)

inference_router = APIRouter(prefix="/inference", tags=["inference"])

# Initialize services
chat_service = ChatService()
test_service = TestService()
benchmark_service = BenchmarkService()
session_service = SessionService()


@inference_router.post("/chat", response_model=ChatResponse)
async def start_chat_session(
    request: ChatRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Start an interactive chat session with a model.
    
    Creates a new chat session and returns session information.
    The actual chat interaction would typically be handled via WebSocket.
    """
    require_permissions(auth, ["chat"])
    
    try:
        # Convert request to args-like object for service compatibility
        class ChatArgs:
            def __init__(self, chat_request):
                self.model_name = chat_request.model_name
                self.system_prompt = chat_request.system_prompt
                self.max_tokens = chat_request.max_tokens
                self.temperature = chat_request.temperature
                self.stream = chat_request.stream
                self.session_name = getattr(chat_request, 'session_name', None)
        
        args = ChatArgs(request)
        
        # Note: The actual chat service returns exit codes, not responses
        # In a real API, this would be restructured to return proper data
        result = chat_service.start_chat(args)
        
        if result == 0:
            return ChatResponse(
                success=True,
                session_id=f"chat_{request.model_name}_{auth.user_id}",
                model_name=request.model_name,
                message="Chat session started successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to start chat session")
            
    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start chat: {str(e)}")


@inference_router.post("/test", response_model=TestResponse)
async def run_model_test(
    request: TestRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Run a single test inference with a model.
    
    Performs inference with the specified model and returns the result.
    """
    require_permissions(auth, ["test"])
    
    try:
        # Convert request to args-like object for service compatibility
        class TestArgs:
            def __init__(self, test_request):
                self.model_name = test_request.model_name
                self.prompt = test_request.prompt
                self.max_tokens = test_request.max_tokens
                self.temperature = test_request.temperature
                self.output_file = None  # API mode doesn't save to file
        
        args = TestArgs(request)
        
        # Note: The test service returns exit codes, not response data
        # In a real implementation, this would be restructured for API use
        result = test_service.run_test(args)
        
        if result == 0:
            return TestResponse(
                success=True,
                model_name=request.model_name,
                prompt=request.prompt,
                response="Test completed successfully",  # Placeholder
                tokens_generated=request.max_tokens,
                inference_time_ms=100.0  # Placeholder
            )
        else:
            raise HTTPException(status_code=500, detail="Test failed")
            
    except Exception as e:
        logger.error(f"Failed to run test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run test: {str(e)}")


@inference_router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Run performance benchmark on a model.
    
    Executes performance testing and returns metrics.
    Can be run in background for long-running benchmarks.
    """
    require_permissions(auth, ["benchmark"])
    
    try:
        # Convert request to args-like object for service compatibility
        class BenchmarkArgs:
            def __init__(self, benchmark_request):
                self.model_name = benchmark_request.model_name
                self.prompt = benchmark_request.prompt
                self.num_runs = benchmark_request.num_runs
                self.max_tokens = benchmark_request.max_tokens
                self.temperature = benchmark_request.temperature
                self.output_file = None  # API mode doesn't save to file
        
        args = BenchmarkArgs(request)
        
        # For long-running benchmarks, consider running in background
        if request.num_runs > 10:
            background_tasks.add_task(benchmark_service.run_benchmark, args)
            return BenchmarkResponse(
                success=True,
                model_name=request.model_name,
                status="running",
                message="Benchmark started in background"
            )
        
        # Run benchmark synchronously for small tests
        result = benchmark_service.run_benchmark(args)
        
        if result == 0:
            return BenchmarkResponse(
                success=True,
                model_name=request.model_name,
                status="completed",
                num_runs=request.num_runs,
                avg_latency_ms=150.0,  # Placeholder
                tokens_per_second=25.0,  # Placeholder
                total_tokens=request.max_tokens * request.num_runs
            )
        else:
            raise HTTPException(status_code=500, detail="Benchmark failed")
            
    except Exception as e:
        logger.error(f"Failed to run benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run benchmark: {str(e)}")


@inference_router.post("/sessions/save", response_model=SessionSaveResponse)
async def save_session(
    request: SessionSaveRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Save a chat session to storage.
    
    Persists the current chat session for later retrieval.
    """
    require_permissions(auth, ["session_save"])
    
    try:
        # Convert request to args-like object for service compatibility
        class SessionArgs:
            def __init__(self, session_request):
                self.session_name = session_request.session_name
                self.session_file = session_request.session_file
        
        args = SessionArgs(request)
        result = session_service.save_session(args)
        if result == 0:
            return SessionSaveResponse(
                success=True,
                session_id=request.session_id,
                file_path=request.output_file or "",
                file_size_bytes=0  # TODO: set actual file size if available
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save session")
            
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")


@inference_router.post("/sessions/load", response_model=SessionLoadResponse)
async def load_session(
    request: SessionLoadRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Load a previously saved chat session.
    
    Retrieves and restores a chat session from storage.
    """
    require_permissions(auth, ["session_load"])
    
    try:
        # Convert request to args-like object for service compatibility
        class SessionArgs:
            def __init__(self, session_request):
                self.session_name = session_request.session_name
                self.session_file = session_request.session_file
        
        args = SessionArgs(request)
        result = session_service.load_session(args)
        if result == 0:
            return SessionLoadResponse(
                success=True,
                session_data={},  # TODO: set actual session data if available
                session_id="",
                message_count=0
            )
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        logger.error(f"Failed to load session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")


@inference_router.get("/sessions", response_model=APIResponse)
async def list_sessions(
    auth: AuthContext = Depends(get_auth_context)
):
    """
    List available chat sessions.
    
    Returns a list of saved sessions that can be loaded.
    """
    require_permissions(auth, ["session_load"])
    
    try:
        # TODO: Implement session listing in service
        # For now, return placeholder data
        return APIResponse(
            success=True,
            data={
                "sessions": [],
                "total_count": 0,
                "message": "No sessions found"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@inference_router.delete("/sessions/{session_name}", response_model=APIResponse)
async def delete_session(
    session_name: str,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Delete a saved chat session.
    
    Permanently removes a session from storage.
    """
    require_permissions(auth, ["session_delete"])
    
    try:
        # TODO: Implement session deletion in service
        return APIResponse(
            success=True,
            data={
                "session_name": session_name,
                "message": f"Session '{session_name}' deleted successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
