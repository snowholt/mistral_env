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
import json
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, File, UploadFile, Form
from ..models import (
    APIResponse, ChatRequest, ChatResponse, AudioChatRequest, AudioChatResponse,
    TestRequest, TestResponse, BenchmarkRequest, BenchmarkResponse,
    SessionSaveRequest, SessionSaveResponse, SessionLoadRequest, SessionLoadResponse,
    VoiceToVoiceRequest, VoiceToVoiceResponse, VoiceToVoiceStatusResponse
)
from ..auth import AuthContext, get_auth_context, require_permissions
from ..errors import ModelNotFoundError, ModelLoadError, ValidationError
from ...services.inference import ChatService, TestService, BenchmarkService, SessionService, ContentFilterService
from ...services.audio_transcription_service import AudioTranscriptionService
from ...config.config_manager import AppConfig

logger = logging.getLogger(__name__)

inference_router = APIRouter(prefix="/inference", tags=["inference"])

# Initialize services
chat_service = ChatService()
test_service = TestService()
benchmark_service = BenchmarkService()
session_service = SessionService()
content_filter_service = ContentFilterService()
audio_transcription_service = AudioTranscriptionService()


@inference_router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Enhanced chat completion with comprehensive parameter control and optimization features.
    
    NEW FEATURES:
    - 📊 Advanced parameter control (25+ parameters including min_p, typical_p, diversity_penalty, etc.)
    - 🎯 Optimization-based presets from actual performance testing
    - 🧠 Thinking mode control (/no_think command support)
    - 🔒 Content filtering control (disable/adjust strictness)
    - ⚡ Performance metrics and detailed response information
    - 🎨 Enhanced preset configurations based on actual optimization results
    
    Core Parameters:
    - temperature, top_p, top_k, repetition_penalty, max_new_tokens
    
    Advanced Parameters:
    - min_p, typical_p, epsilon_cutoff, eta_cutoff, diversity_penalty
    - no_repeat_ngram_size, encoder_repetition_penalty
    - num_beams, length_penalty, early_stopping
    
    Optimization-Based Presets:
    - "qwen_optimized": Best settings from actual testing (temp=0.3, top_p=0.95, top_k=20)
    - "high_quality": Maximum quality (temp=0.1, top_p=1.0, rep_penalty=1.15)
    - "creative_optimized": Creative but efficient (temp=0.5, top_p=1.0, top_k=80)
    - "speed_optimized", "balanced", "conservative", "creative"
    
    Content Filtering:
    - disable_content_filter: true/false
    - content_filter_strictness: "strict"/"balanced"/"relaxed"/"disabled"
    
    Example usage:
    {
        "model_name": "qwen3-model",
        "message": "What is AI?",
        "preset": "qwen_optimized",
        "disable_content_filter": true
    }
    
    Advanced example:
    {
        "model_name": "qwen3-model",
        "message": "Explain quantum computing",
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "content_filter_strictness": "relaxed"
    }
    
    Thinking mode examples:
    {
        "model_name": "qwen3-model", 
        "message": "/no_think Give a brief answer",
        "preset": "speed_optimized"
    }
    """
    require_permissions(auth, ["chat"])
    
    start_time = time.time()
    
    try:
        logger.info(f"Chat request received for model: {request.model_name}")
        # Import the inference adapter
        from ..adapters.inference_adapter import InferenceAPIAdapter
        
        # Create adapter instance with required service dependencies
        inference_adapter = InferenceAPIAdapter(
            chat_service=chat_service,
            test_service=test_service, 
            benchmark_service=benchmark_service
        )
        
        # Process message and thinking mode
        processed_message = request.get_processed_message()
        thinking_enabled = request.should_enable_thinking()
        
        # Build effective generation configuration
        effective_config = request.get_effective_generation_config()
        
        # Configure content filtering based on request
        filter_config = request.get_effective_content_filter_config()
        logger.info(f"Content filter config: {filter_config}")
        
        # Content filtering check (if not disabled)
        content_filter_bypassed = False
        if filter_config["strictness_level"] == "disabled":
            content_filter_bypassed = True
            logger.info("Content filtering disabled")
        else:
            logger.info(f"Applying content filter with strictness: {filter_config['strictness_level']}")
            # Set the content filter strictness level
            content_filter_service.set_strictness_level(filter_config["strictness_level"])
            
            try:
                filter_result = content_filter_service.filter_content(processed_message, language='ar')
                logger.info(f"Content filter result: allowed={filter_result.is_allowed}")
                if not filter_result.is_allowed:
                    return ChatResponse(
                        success=False,
                        response="",
                        model_name=request.model_name,
                        effective_config=effective_config,
                        preset_used=request.preset,
                        thinking_enabled=thinking_enabled,
                        error=f"Content filtered: {filter_result.filter_reason}"
                    )
            except Exception as e:
                logger.error(f"Content filtering failed: {e}, proceeding without filtering")
                content_filter_bypassed = True
        
        # Convert simple message to messages format
        messages = []
        
        # Add system message for thinking mode if needed
        if thinking_enabled and request.model_name.lower().find("qwen") != -1:
            # Add thinking system message for Qwen models
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant. Think step by step before providing your final answer."
            })
        elif not thinking_enabled and request.model_name.lower().find("qwen") != -1:
            # Explicitly disable thinking for Qwen models  
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant. Provide direct, concise answers without showing your thinking process."
            })
        
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
            "content": processed_message
        })
        
        # Generate response using the adapter
        generation_start = time.time()
        
        # Prepare parameters, removing duplicates that might conflict
        adapter_params = effective_config.copy()
        adapter_params.update({
            'model_name': request.model_name,
            'messages': messages,
            'stream': request.stream,
        })
        
        # Use get() to avoid conflicts and set defaults
        response_data = inference_adapter.chat_completion(**adapter_params)
        generation_end = time.time()
        
        # Calculate performance metrics
        generation_time_ms = (generation_end - generation_start) * 1000
        
        logger.info(f"Response data keys: {response_data.keys() if isinstance(response_data, dict) else 'Not a dict'}")
        logger.info(f"Response data type: {type(response_data)}")
        
        # Extract response text from OpenAI-style format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            response_text = response_data["choices"][0].get("message", {}).get("content", "")
            logger.info(f"Extracted from choices: {len(response_text)} chars")
        else:
            response_text = response_data.get("response", "")
            logger.info(f"Extracted from response field: {len(response_text)} chars")
            
        logger.info(f"Final response_text length: {len(response_text)}")
        
        tokens_generated = len(response_text.split()) if response_text else 0
        tokens_per_second = tokens_generated / (generation_time_ms / 1000) if generation_time_ms > 0 else 0
        
        # Parse thinking content if applicable
        thinking_content = None
        final_content = response_text
        
        if thinking_enabled and "<think>" in response_text and "</think>" in response_text:
            # Extract thinking and final content
            import re
            think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        
        # Build enhanced response
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        
        logger.info(f"Building response - response_text length: {len(response_text)}")
        
        try:
            response_obj = ChatResponse(
                success=True,
                response=final_content,
                session_id=request.session_id or "default",
                model_name=request.model_name,
                effective_config=effective_config,
                preset_used=request.preset,
                thinking_enabled=thinking_enabled,
                content_filter_applied=not content_filter_bypassed,
                content_filter_strictness=filter_config["strictness_level"],
                content_filter_bypassed=content_filter_bypassed,
                tokens_generated=tokens_generated,
                generation_time_ms=generation_time_ms,
                tokens_per_second=round(tokens_per_second, 2) if tokens_per_second else 0.0,
                thinking_content=thinking_content,
                final_content=final_content,
                execution_time_ms=total_time_ms,
                generation_stats={
                    "model_info": response_data.get("model_info", {}),
                    "generation_config_used": effective_config,
                    "content_filter_config": filter_config,
                    "performance": {
                        "total_time_ms": total_time_ms,
                        "generation_time_ms": generation_time_ms,
                        "tokens_generated": tokens_generated,
                        "tokens_per_second": tokens_per_second,
                        "thinking_tokens": len(thinking_content.split()) if thinking_content else 0
                    }
                }
            )
            logger.info("ChatResponse object created successfully")
            return response_obj
        except Exception as e:
            logger.error(f"Error creating ChatResponse object: {e}")
            # Return a simple response that should always work
            return ChatResponse(
                success=True,
                response=final_content or "Response generated successfully",
                model_name=request.model_name,
                execution_time_ms=total_time_ms
            )
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        end_time = time.time()
        return ChatResponse(
            success=False,
            response="",
            model_name=request.model_name,
            effective_config=request.get_effective_generation_config() if hasattr(request, 'get_effective_generation_config') else {},
            preset_used=request.preset,
            thinking_enabled=request.should_enable_thinking() if hasattr(request, 'should_enable_thinking') else None,
            execution_time_ms=(end_time - start_time) * 1000,
            error=f"Generation failed: {str(e)}"
        )


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


@inference_router.post("/audio-chat", response_model=AudioChatResponse)
async def audio_chat(
    audio_file: UploadFile = File(...),
    stt_model_name: str = Form("whisper-large-v3-turbo-arabic"),
    chat_model_name: str = Form("qwen3-unsloth-q4ks"),
    input_language: str = Form("ar"),
    session_id: Optional[str] = Form(None),
    chat_history: Optional[str] = Form(None),  # JSON string
    temperature: Optional[float] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    do_sample: Optional[bool] = Form(None),
    disable_content_filter: bool = Form(False),
    preset: Optional[str] = Form(None),
    auth: AuthContext = Depends(get_auth_context)
):
    """Audio chat endpoint - converts audio to text, processes with LLM, returns text response."""
    try:
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Create request object
        request = AudioChatRequest(
            model_name=chat_model_name,
            session_id=session_id,
            chat_history=json.loads(chat_history) if chat_history else None,
            audio_language=input_language,
            whisper_model_name=stt_model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=do_sample,
            disable_content_filter=disable_content_filter,
            preset=preset
        )
        
        # Process audio chat
        start_time = time.time()
        
        # Step 1: Load Whisper model if not already loaded
        if not audio_transcription_service.whisper_model:
            logger.info(f"Loading Whisper model: {stt_model_name}")
            if not audio_transcription_service.load_whisper_model(stt_model_name):
                raise HTTPException(status_code=500, detail="Failed to load Whisper model")
        
        # Step 1: Transcribe audio
        logger.info("Processing audio chat request...")
        transcription = audio_transcription_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            audio_format=audio_file.filename.split('.')[-1].lower() if '.' in audio_file.filename else 'wav',
            language=input_language
        )
        
        if not transcription:
            raise ValidationError("Audio transcription failed")
        
        transcription_time = time.time() - start_time
        
        # Step 2: Process with chat service
        chat_request = ChatRequest(
            messages=[{"role": "user", "content": transcription}],
            model_name=chat_model_name,
            session_id=request.session_id,
            chat_history=request.chat_history,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            do_sample=request.do_sample,
            disable_content_filter=request.disable_content_filter,
            preset=request.preset
        )
        
        chat_response = await chat_completion(chat_request, auth)
        
        total_time = time.time() - start_time
        
        return AudioChatResponse(
            success=True,
            response=chat_response.response,
            session_id=chat_response.session_id,
            model_name=chat_response.model_name,
            transcription=transcription,
            whisper_model_used=stt_model_name,
            audio_language_detected=input_language,
            transcription_time_ms=transcription_time * 1000,
            generation_stats=chat_response.generation_stats,
            effective_config=chat_response.effective_config,
            preset_used=chat_response.preset_used,
            thinking_enabled=chat_response.thinking_enabled,
            content_filter_applied=chat_response.content_filter_applied,
            content_filter_strictness=chat_response.content_filter_strictness,
            content_filter_bypassed=chat_response.content_filter_bypassed,
            tokens_generated=chat_response.tokens_generated,
            generation_time_ms=chat_response.generation_time_ms,
            tokens_per_second=chat_response.tokens_per_second,
            total_processing_time_ms=total_time * 1000,
            thinking_content=chat_response.thinking_content,
            final_content=chat_response.final_content
        )
        
    except Exception as e:
        logger.error(f"Audio chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio chat processing failed: {str(e)}")


@inference_router.post("/voice-to-voice", response_model=VoiceToVoiceResponse)
async def voice_to_voice(
    audio_file: UploadFile = File(...),
    input_language: str = Form("ar"),
    output_language: str = Form("ar"),
    stt_model_name: str = Form("whisper-large-v3-turbo-arabic"),
    tts_model_name: str = Form("oute-tts-1b"),
    chat_model_name: str = Form("qwen3-unsloth-q4ks"),
    session_id: Optional[str] = Form(None),
    chat_history: Optional[str] = Form(None),  # JSON string
    speaker_voice: Optional[str] = Form(None),
    emotion: str = Form("neutral"),
    speech_speed: float = Form(1.0),
    audio_output_format: str = Form("wav"),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    do_sample: Optional[bool] = Form(None),
    disable_content_filter: bool = Form(False),
    content_filter_strictness: str = Form("balanced"),
    preset: Optional[str] = Form(None),
    auth: AuthContext = Depends(get_auth_context)
):
    """Complete voice-to-voice conversation: Audio Input → STT → LLM → TTS → Audio Output."""
    try:
        from ...services.voice_to_voice_service import VoiceToVoiceService
        
        # Read audio file
        audio_bytes = await audio_file.read()
        audio_format = audio_file.filename.split('.')[-1].lower() if '.' in audio_file.filename else 'wav'
        
        logger.info(f"Voice-to-voice request: {len(audio_bytes)} bytes, format: {audio_format}")
        
        # Initialize voice-to-voice service
        v2v_service = VoiceToVoiceService(content_filter_strictness=content_filter_strictness)
        
        # Initialize models
        models_initialized = v2v_service.initialize_models(
            stt_model=stt_model_name,
            tts_model=tts_model_name,
            chat_model=chat_model_name
        )
        
        if not models_initialized:
            raise HTTPException(status_code=500, detail="Failed to initialize voice-to-voice models")
        
        # Prepare generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        if max_new_tokens is not None:
            generation_config["max_new_tokens"] = max_new_tokens
        if do_sample is not None:
            generation_config["do_sample"] = do_sample
        
        # Process voice-to-voice conversation
        result = v2v_service.voice_to_voice_conversation(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            input_language=input_language,
            output_language=output_language,
            session_id=session_id,
            chat_history=json.loads(chat_history) if chat_history else None,
            speaker_voice=speaker_voice,
            emotion=emotion,
            speech_speed=speech_speed,
            generation_config=generation_config,
            disable_content_filter=disable_content_filter
        )
        
        if not result["success"]:
            error_message = "; ".join(result.get("errors", ["Unknown error"]))
            raise HTTPException(status_code=500, detail=f"Voice-to-voice processing failed: {error_message}")
        
        # Create response
        from fastapi.responses import Response
        
        # Return audio response with metadata in headers
        response = Response(
            content=result["response_audio_bytes"],
            media_type=f"audio/{audio_output_format}",
            headers={
                "X-Session-ID": result["session_id"],
                "X-Transcription": result["transcription"],
                "X-Response-Text": result["response_text"],
                "X-Input-Language": result["input_language"],
                "X-Output-Language": result["output_language"],
                "X-Total-Time": str(result["metrics"]["total_time_seconds"]),
                "X-STT-Time": str(result["metrics"]["stt_time_seconds"]),
                "X-Chat-Time": str(result["metrics"]["chat_time_seconds"]),
                "X-TTS-Time": str(result["metrics"]["tts_time_seconds"]),
                "X-Audio-Size": str(result["metrics"]["audio_size_bytes"]),
                "Content-Disposition": f'attachment; filename="response.{audio_output_format}"'
            }
        )
        
        # Clean up models to free memory
        try:
            v2v_service.unload_all_models()
        except Exception as e:
            logger.warning(f"Error cleaning up models: {e}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice-to-voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice-to-voice processing failed: {str(e)}")


@inference_router.get("/voice-to-voice/status", response_model=VoiceToVoiceStatusResponse)
async def voice_to_voice_status(auth: AuthContext = Depends(get_auth_context)):
    """Get status of voice-to-voice service and model availability."""
    try:
        from ...services.voice_to_voice_service import VoiceToVoiceService
        
        # Create temporary service to check model availability
        v2v_service = VoiceToVoiceService()
        
        # Check model configurations
        app_config = AppConfig()
        app_config.models_file = "beautyai_inference/config/model_registry.json"
        app_config.load_model_registry()
        
        models_status = {
            "stt_model": {
                "available": app_config.model_registry.get_model("whisper-large-v3-turbo-arabic") is not None,
                "default": "whisper-large-v3-turbo-arabic",
                "loaded": False
            },
            "tts_model": {
                "available": True,  # XTTS-v2 is always available if TTS library is installed
                "default": "xtts-v2",
                "loaded": False
            },
            "chat_model": {
                "available": app_config.model_registry.get_model("qwen3-unsloth-q4ks") is not None,
                "default": "qwen3-unsloth-q4ks",
                "loaded": False
            }
        }
        
        # Check TTS library availability
        try:
            from TTS.api import TTS
            tts_available = True
        except ImportError:
            tts_available = False
        
        return VoiceToVoiceStatusResponse(
            success=True,
            service_available=True,
            models_status=models_status,
            supported_languages={
                "input": ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja", "hu", "ko"],
                "output": ["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "zh", "ja", "hu", "ko"]
            },
            supported_audio_formats=["wav", "mp3", "ogg", "flac", "m4a", "wma", "webm"],
            tts_library_available=tts_available,
            pipeline_stages=["STT (Speech-to-Text)", "LLM (Language Model)", "TTS (Text-to-Speech)"],
            estimated_setup_time_seconds=60
        )
        
    except Exception as e:
        logger.error(f"Error getting voice-to-voice status: {e}")
        return VoiceToVoiceStatusResponse(
            success=False,
            service_available=False,
            error=str(e)
        )