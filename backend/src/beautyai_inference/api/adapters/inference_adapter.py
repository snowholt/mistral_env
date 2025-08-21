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
from ...services.shared import get_shared_model_manager

logger = logging.getLogger(__name__)


class InferenceAPIAdapter(APIServiceAdapter):
    """
    API adapter for inference operations.
    
    Provides API-compatible interface for:
    - Chat completions (streaming and non-streaming)
    - Model validation
    """
    
    def __init__(self, chat_service: ChatService):
        """Initialize inference API adapter with chat service."""
        self.chat_service = chat_service
        self.model_manager = get_shared_model_manager()
        super().__init__(self.chat_service)  # Use chat as primary service
    
    def get_supported_operations(self) -> Dict[str, str]:
        """Get dictionary of supported operations and their descriptions."""
        return {
            "chat_completion": "Generate chat completion",
            "chat_completion_stream": "Generate streaming chat completion",
            "validate_input": "Validate inference input parameters"
        }
    
    def chat_completion(self, model_name: str, messages: List[Dict[str, str]], 
                             max_tokens: Optional[int] = None, 
                             temperature: float = 0.7,
                             stream: bool = False,
                             disable_content_filter: bool = False,
                             **kwargs) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate chat completion.
        
        Args:
            model_name: Name of the model to use
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            disable_content_filter: Whether to bypass content filtering
            **kwargs: Additional generation parameters
            
        Returns:
            Chat completion response or stream iterator
        """
        start_time = time.time()
        
        try:
            # No need to configure chat service - it's already initialized
            
            if stream:
                return self._stream_chat_completion(
                    model_name, messages, max_tokens, temperature, 
                    disable_content_filter, **kwargs
                )
            else:
                return self._complete_chat_completion(
                    model_name, messages, max_tokens, temperature, 
                    disable_content_filter, **kwargs
                )
                
        except Exception as e:
            logger.error(f"Failed to generate chat completion: {e}")
            raise InferenceError(f"Chat completion failed: {e}")
    
    def _complete_chat_completion(self, model_name: str, messages: List[Dict[str, str]], 
                                       max_tokens: Optional[int], temperature: float, 
                                       disable_content_filter: bool = False,
                                       **kwargs) -> Dict[str, Any]:
        """Generate non-streaming chat completion."""
        logger.info(f"Starting chat completion for model: {model_name}")
        
        # Extract the user message from the messages list
        user_message = ""
        conversation_history = []
        
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
            elif msg.get("role") == "assistant":
                conversation_history.append(msg)
        
        if not user_message:
            user_message = "Hello"  # Fallback message
        
        # Get model configuration from registry
        from ...config.config_manager import AppConfig
        from ...services.model.registry_service import ModelRegistryService
        from pathlib import Path
        
        # Load app config with model registry
        config_dir = Path(__file__).parent.parent.parent / "config"
        model_registry_path = config_dir / "model_registry.json"
        app_config = AppConfig()
        app_config.models_file = str(model_registry_path)
        app_config.load_model_registry()
        
        # Get model config using registry service
        registry_service = ModelRegistryService()
        model_config = registry_service.get_model(app_config, model_name)
        if not model_config:
            raise InferenceError(f"Model '{model_name}' not found in registry")
        
        # Prepare generation parameters
        generation_config = {
            'temperature': temperature,
            **kwargs
        }
        
        # Map max_tokens to max_new_tokens for compatibility
        if max_tokens:
            generation_config['max_new_tokens'] = max_tokens
        
        logger.info(f"Chat generation params: {generation_config}")
        
        try:
            # Use the ChatService to handle the chat completion
            response, detected_language, updated_history, session_id = self.chat_service.chat(
                message=user_message,
                model_name=model_name,
                model_config=model_config,
                generation_config=generation_config,
                conversation_history=conversation_history,
                disable_content_filter=disable_content_filter
            )
            logger.info(f"ChatService.chat() completed. Response length: {len(response)} characters")
            
            response_text = response
            
        except Exception as e:
            logger.error(f"Error during chat generation: {e}")
            raise InferenceError(f"Chat generation failed: {e}")
        
        if not response_text:
            logger.warning("ChatService returned empty response")
            response_text = "I apologize, but I couldn't generate a response. Please try again."
        
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
                               disable_content_filter: bool = False,
                               **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate streaming chat completion."""
        # For now, fall back to non-streaming and yield as a single chunk
        # TODO: Implement proper streaming support
        
        # Extract the user message from the messages list
        user_message = ""
        conversation_history = []
        
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
            elif msg.get("role") == "assistant":
                conversation_history.append(msg)
        
        if not user_message:
            user_message = "Hello"  # Fallback message
        
        # Get model configuration from registry
        from ...config.config_manager import AppConfig
        from ...services.model.registry_service import ModelRegistryService
        from pathlib import Path
        
        # Load app config with model registry
        config_dir = Path(__file__).parent.parent.parent / "config"
        model_registry_path = config_dir / "model_registry.json"
        app_config = AppConfig()
        app_config.models_file = str(model_registry_path)
        app_config.load_model_registry()
        
        # Get model config using registry service
        registry_service = ModelRegistryService()
        model_config = registry_service.get_model(app_config, model_name)
        if not model_config:
            raise InferenceError(f"Model '{model_name}' not found in registry")
        
        # Prepare generation parameters
        generation_config = {
            'temperature': temperature,
            **kwargs
        }
        
        # Map max_tokens to max_new_tokens for compatibility
        if max_tokens:
            generation_config['max_new_tokens'] = max_tokens
        
        # Generate response using the ChatService
        response, detected_language, updated_history, session_id = self.chat_service.chat(
            message=user_message,
            model_name=model_name,
            model_config=model_config,
            generation_config=generation_config,
            conversation_history=conversation_history,
            disable_content_filter=disable_content_filter
        )
        
        # Yield the response as a single chunk
        yield {
            "choices": [{
                "delta": {
                    "content": response
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
