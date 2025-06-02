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
        # Import the inference adapter
        from ..adapters.inference_adapter import InferenceAPIAdapter
        
        # Create adapter instance with required service dependencies
        inference_adapter = InferenceAPIAdapter(
            chat_service=chat_service,
            test_service=test_service, 
            benchmark_service=benchmark_service
        )
        
        # Extract generation parameters from generation_config
        generation_params = {}
        if request.generation_config:
            generation_params.update(request.generation_config)
        
        # Create messages format for the prompt
        messages = [{"role": "user", "content": request.prompt}]
        
        # Generate response using the adapter
        response_data = await inference_adapter.chat_completion(
            model_name=request.model_name,
            messages=messages,
            stream=False,
            **generation_params
        )
        
        # Extract response text
        if "choices" in response_data and len(response_data["choices"]) > 0:
            response_text = response_data["choices"][0].get("message", {}).get("content", "")
        else:
            response_text = "No response generated"
        
        return TestResponse(
            success=True,
            model_name=request.model_name,
            prompt=request.prompt,
            response=response_text,
            generation_stats=response_data.get("usage", {})
        )
            
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
        # Import the inference adapter
        from ..adapters.inference_adapter import InferenceAPIAdapter
        
        # Create adapter instance with required service dependencies
        inference_adapter = InferenceAPIAdapter(
            chat_service=chat_service,
            test_service=test_service, 
            benchmark_service=benchmark_service
        )
        
        # Extract configuration parameters
        config = request.config or {}
        num_runs = config.get("num_runs", 5)
        prompt = config.get("prompt", "Hello, how are you?")
        
        # For large benchmarks, run in background
        if num_runs > 10:
            background_tasks.add_task(_run_benchmark_task, request, inference_adapter)
            return BenchmarkResponse(
                success=True,
                model_name=request.model_name,
                benchmark_type=request.benchmark_type,
                summary={"status": "running", "message": "Benchmark started in background"}
            )
        
        # Run benchmark synchronously for small tests
        start_time = time.time()
        total_tokens = 0
        responses = []
        
        for i in range(num_runs):
            messages = [{"role": "user", "content": f"{prompt} (run {i+1})"}]
            
            response_data = await inference_adapter.chat_completion(
                model_name=request.model_name,
                messages=messages,
                stream=False,
                **config.get("generation_config", {})
            )
            
            responses.append(response_data)
            if "usage" in response_data:
                total_tokens += response_data["usage"].get("total_tokens", 0)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_latency = (total_time / num_runs) * 1000  # Convert to ms
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return BenchmarkResponse(
            success=True,
            model_name=request.model_name,
            benchmark_type=request.benchmark_type,
            results={
                "num_runs": num_runs,
                "total_time_s": total_time,
                "avg_latency_ms": avg_latency,
                "tokens_per_second": tokens_per_second,
                "total_tokens": total_tokens
            },
            summary={
                "status": "completed",
                "avg_latency_ms": avg_latency,
                "tokens_per_second": tokens_per_second
            }
        )
            
    except Exception as e:
        logger.error(f"Failed to run benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run benchmark: {str(e)}")

async def _run_benchmark_task(request: BenchmarkRequest, inference_adapter):
    """Background task for running large benchmarks."""
    # This would be implemented for background processing
    pass


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
        # Create sessions directory if it doesn't exist
        import os
        sessions_dir = "/home/lumi/beautyai/sessions"
        os.makedirs(sessions_dir, exist_ok=True)
        
        # Use output_file if provided, otherwise generate filename
        if request.output_file:
            file_path = request.output_file
        else:
            file_path = os.path.join(sessions_dir, f"{request.session_id}.json")
        
        # Save session data to file
        import json
        with open(file_path, 'w') as f:
            json.dump(request.session_data, f, indent=2)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return SessionSaveResponse(
            success=True,
            session_id=request.session_id,
            file_path=file_path,
            file_size_bytes=file_size
        )
            
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
        import json
        import os
        
        # Check if file exists
        if not os.path.exists(request.input_file):
            raise HTTPException(status_code=404, detail=f"Session file not found: {request.input_file}")
        
        # Load session data from file
        with open(request.input_file, 'r') as f:
            session_data = json.load(f)
        
        # Extract session info
        session_id = session_data.get("session_id", "")
        messages = session_data.get("messages", [])
        message_count = len(messages)
        
        return SessionLoadResponse(
            success=True,
            session_data=session_data,
            session_id=session_id,
            message_count=message_count
        )
            
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
        import os
        import json
        
        sessions_dir = "/home/lumi/beautyai/sessions"
        sessions = []
        
        # Check if sessions directory exists
        if os.path.exists(sessions_dir):
            # List all JSON files in sessions directory
            for filename in os.listdir(sessions_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(sessions_dir, filename)
                    try:
                        # Try to load session metadata
                        with open(file_path, 'r') as f:
                            session_data = json.load(f)
                        
                        sessions.append({
                            "session_id": session_data.get("session_id", filename[:-5]),  # Remove .json
                            "file_path": file_path,
                            "file_size": os.path.getsize(file_path),
                            "message_count": len(session_data.get("messages", [])),
                            "last_modified": os.path.getmtime(file_path)
                        })
                    except Exception as e:
                        logger.warning(f"Could not read session file {filename}: {e}")
        
        return APIResponse(
            success=True,
            data={
                "sessions": sessions,
                "total_count": len(sessions),
                "message": f"Found {len(sessions)} sessions"
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
        import os
        
        sessions_dir = "/home/lumi/beautyai/sessions"
        file_path = os.path.join(sessions_dir, f"{session_name}.json")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Session not found: {session_name}")
        
        # Delete the file
        os.remove(file_path)
        
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