"""
Inference API Adapter.

Provides API-compatible interface for running inference operations,
bridging inference services with REST/GraphQL endpoints.
"""
from typing import Dict, Any, Optional, List, Union, Iterator
import logging
import time

from .base_adapter import APIServiceAdapter
from ..models import APIRequest, APIResponse
from ..errors import ModelNotFoundError, ValidationError, InferenceError
from ...services.inference.chat_service import ChatService
from ...services.inference.test_service import TestService
from ...services.inference.benchmark_service import BenchmarkService

logger = logging.getLogger(__name__)


class InferenceAPIAdapter(APIServiceAdapter):
    """
    API adapter for inference operations.
    
    Provides API-compatible interface for:
    - Chat completions (streaming and non-streaming)
    - Model testing and validation
    - Performance benchmarking
    """
    
    def __init__(self, chat_service: ChatService, 
                 test_service: TestService,
                 benchmark_service: BenchmarkService):
        """Initialize inference API adapter with required services."""
        self.chat_service = chat_service
        self.test_service = test_service
        self.benchmark_service = benchmark_service
        super().__init__(self.chat_service)  # Use chat as primary service
    
    def get_supported_operations(self) -> Dict[str, str]:
        """Get dictionary of supported operations and their descriptions."""
        return {
            "chat_completion": "Generate chat completion",
            "chat_completion_stream": "Generate streaming chat completion",
            "test_model": "Test model with predefined prompts",
            "benchmark_model": "Run performance benchmarks",
            "validate_input": "Validate inference input parameters"
        }
    
    async def chat_completion(self, model_name: str, messages: List[Dict[str, str]], 
                             max_tokens: Optional[int] = None, 
                             temperature: float = 0.7,
                             stream: bool = False,
                             **kwargs) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate chat completion.
        
        Args:
            model_name: Name of the model to use
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            Chat completion response or stream iterator
        """
        start_time = time.time()
        
        try:
            # Configure services with default config
            self.chat_service.configure({})
            
            if stream:
                return self._stream_chat_completion(
                    model_name, messages, max_tokens, temperature, **kwargs
                )
            else:
                return await self._complete_chat_completion(
                    model_name, messages, max_tokens, temperature, **kwargs
                )
                
        except Exception as e:
            logger.error(f"Failed to generate chat completion: {e}")
            raise InferenceError(f"Chat completion failed: {e}")
    
    async def _complete_chat_completion(self, model_name: str, messages: List[Dict[str, str]], 
                                       max_tokens: Optional[int], temperature: float, 
                                       **kwargs) -> Dict[str, Any]:
        """Generate non-streaming chat completion."""
        from ...core.model_manager import ModelManager
        
        # Get the loaded model
        model_manager = ModelManager()
        model = model_manager.get_loaded_model(model_name)
        
        if model is None:
            raise InferenceError(f"Model '{model_name}' is not loaded")
        
        # Prepare generation parameters
        generation_params = {
            'temperature': temperature,
            **kwargs
        }
        if max_tokens:
            generation_params['max_tokens'] = max_tokens
        
        # Generate response using the model's chat method
        response_text = model.chat(messages, **generation_params)
        
        # Count tokens (rough estimation)
        prompt_text = self._messages_to_prompt(messages)
        prompt_tokens = len(prompt_text.split()) 
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "model": model_name,
            "created": int(time.time())
        }
    
    def _stream_chat_completion(self, model_name: str, messages: List[Dict[str, str]], 
                               max_tokens: Optional[int], temperature: float, 
                               **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate streaming chat completion."""
        # For now, fall back to non-streaming and yield as a single chunk
        # TODO: Implement proper streaming support
        from ...core.model_manager import ModelManager
        
        # Get the loaded model
        model_manager = ModelManager()
        model = model_manager.get_loaded_model(model_name)
        
        if model is None:
            raise InferenceError(f"Model '{model_name}' is not loaded")
        
        # Prepare generation parameters
        generation_params = {
            'temperature': temperature,
            **kwargs
        }
        if max_tokens:
            generation_params['max_tokens'] = max_tokens
        
        # Generate response using the model's chat method
        response_text = model.chat(messages, **generation_params)
        
        # Yield the response as a single chunk
        yield {
            "choices": [{
                "delta": {
                    "content": response_text
                },
                "index": 0,
                "finish_reason": "stop"
            }],
            "model": model_name,
            "created": int(time.time())
        }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    async def test_model(self, model_name: str, test_type: str = "basic",
                        custom_prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test model with predefined or custom prompts.
        
        Args:
            model_name: Name of the model to test
            test_type: Type of test to run (basic, advanced, custom)
            custom_prompts: Custom prompts for testing
            
        Returns:
            Dictionary with test results
        """
        try:
            # Configure services
            self.test_service.configure({})
            
            # Run the test
            result = await self.test_service.test_model(
                model_name=model_name,
                test_type=test_type,
                custom_prompts=custom_prompts
            )
            
            return {
                "model_name": model_name,
                "test_type": test_type,
                "results": result.get("test_results", []),
                "summary": result.get("summary", {}),
                "passed": result.get("passed", False),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to test model '{model_name}': {e}")
            raise InferenceError(f"Model testing failed: {e}")
    
    async def benchmark_model(self, model_name: str, benchmark_type: str = "latency",
                             num_requests: int = 10, concurrent: bool = False) -> Dict[str, Any]:
        """
        Run performance benchmarks on a model.
        
        Args:
            model_name: Name of the model to benchmark
            benchmark_type: Type of benchmark (latency, throughput, memory)
            num_requests: Number of requests to run
            concurrent: Whether to run requests concurrently
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            # Configure services
            self.benchmark_service.configure({})
            
            # Run the benchmark
            result = await self.benchmark_service.run_benchmark(
                model_name=model_name,
                benchmark_type=benchmark_type,
                num_requests=num_requests,
                concurrent=concurrent
            )
            
            return {
                "model_name": model_name,
                "benchmark_type": benchmark_type,
                "num_requests": num_requests,
                "concurrent": concurrent,
                "metrics": result.get("metrics", {}),
                "statistics": result.get("statistics", {}),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to benchmark model '{model_name}': {e}")
            raise InferenceError(f"Model benchmarking failed: {e}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate inference input parameters.
        
        Args:
            input_data: Input parameters to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_errors = []
        warnings = []
        
        # Validate required fields
        if "model_name" not in input_data:
            validation_errors.append("model_name is required")
        
        if "messages" in input_data:
            messages = input_data["messages"]
            if not isinstance(messages, list):
                validation_errors.append("messages must be a list")
            else:
                for i, message in enumerate(messages):
                    if not isinstance(message, dict):
                        validation_errors.append(f"messages[{i}] must be a dictionary")
                    elif "content" not in message:
                        validation_errors.append(f"messages[{i}] must have 'content' field")
        
        # Validate optional parameters
        if "max_tokens" in input_data:
            max_tokens = input_data["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                validation_errors.append("max_tokens must be a positive integer")
            elif max_tokens > 4096:
                warnings.append("max_tokens is very high, may impact performance")
        
        if "temperature" in input_data:
            temperature = input_data["temperature"]
            if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
                validation_errors.append("temperature must be between 0 and 2")
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": warnings,
            "timestamp": int(time.time())
        }
