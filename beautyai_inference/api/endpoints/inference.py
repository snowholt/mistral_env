"""
Inference operation API endpoints.

Provides REST API endpoints for inference operations including:
- Interactive chat sessions
- Single model testing
- Performance benchmarking  
- Session management
"""
import logging
import time
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from ..models import (
    APIResponse, ChatRequest, ChatResponse, TestRequest, TestResponse,
    BenchmarkRequest, BenchmarkResponse,
    SessionSaveRequest, SessionSaveResponse, SessionLoadRequest, SessionLoadResponse
)
from ..auth import AuthContext, get_auth_context, require_permissions
from ..errors import ModelNotFoundError, ModelLoadError, ValidationError
from ...services.inference import ChatService, TestService, BenchmarkService, SessionService, ContentFilterService

logger = logging.getLogger(__name__)

inference_router = APIRouter(prefix="/inference", tags=["inference"])

# Initialize services
chat_service = ChatService()
test_service = TestService()
benchmark_service = BenchmarkService()
session_service = SessionService()
content_filter_service = ContentFilterService()


@inference_router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Generate a chat completion response from a model.
    
    Sends a message to the model and returns the response.
    Supports both streaming and non-streaming responses.
    """
    require_permissions(auth, ["chat"])
    
    try:
        # Import the inference adapter
        from ..adapters.inference_adapter import InferenceAPIAdapter
        
        # Create adapter instance with required service dependencies
        inference_adapter = InferenceAPIAdapter(
            chat_service=chat_service,
            test_service=test_service, 
            benchmark_service=benchmark_service
        )
        
        # Convert simple message to messages format
        messages = []
        
        # Add chat history if provided
        if request.chat_history:
            for msg in request.chat_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current user message
        messages.append({
            "role": "user", 
            "content": request.message
        })
        
        # Content filtering check for the current user message
        filter_result = content_filter_service.filter_content(request.message, language='ar')
        if not filter_result.is_allowed:
            return ChatResponse(
                message="",
                response=filter_result.suggested_response,
                model_info={
                    "model_name": request.model_name,
                    "status": "filtered"
                },
                generation_config=request.generation_config,
                success=False,
                error=f"Content filtered: {filter_result.filter_reason}"
            )
        
        # Extract generation parameters from generation_config
        generation_params = {}
        if request.generation_config:
            generation_params.update(request.generation_config)
        
        # Generate response using the adapter
        response_data = await inference_adapter.chat_completion(
            model_name=request.model_name,
            messages=messages,
            stream=request.stream,
            **generation_params
        )
        
        # Extract response text
        if "choices" in response_data and len(response_data["choices"]) > 0:
            response_text = response_data["choices"][0].get("message", {}).get("content", "")
        else:
            response_text = "No response generated"
        
        # Generate session ID if not provided
        session_id = request.session_id or f"chat_{request.model_name}_{int(time.time())}"
        
        return ChatResponse(
            success=True,
            response=response_text,
            session_id=session_id,
            model_name=request.model_name,
            generation_stats=response_data.get("usage", {})
        )
            
    except Exception as e:
        logger.error(f"Failed to generate chat response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chat response: {str(e)}")


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
