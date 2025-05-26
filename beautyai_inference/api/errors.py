"""
API Error Handling Module.

Provides standardized error handling patterns for API endpoints,
including validation, error responses, and exception handling.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import logging
import traceback
from functools import wraps

from .models import ErrorResponse

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base class for all API errors."""
    
    def __init__(self, message: str, error_code: str = "API_ERROR", 
                 details: Optional[Dict[str, Any]] = None, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)
    
    def to_response(self) -> ErrorResponse:
        """Convert to standardized error response."""
        return ErrorResponse(
            error_code=self.error_code,
            error_message=self.message,
            error_details=self.details
        )


class ValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details or {},
            status_code=400
        )
        if field:
            self.details["field"] = field


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details or {},
            status_code=401
        )


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Authorization failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details=details or {},
            status_code=403
        )


class ModelNotFoundError(APIError):
    """Raised when a model is not found."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Model '{model_name}' not found",
            error_code="MODEL_NOT_FOUND",
            details=details or {"model_name": model_name},
            status_code=404
        )


class ModelLoadError(APIError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_ERROR",
            details=details or {"model_name": model_name, "reason": reason},
            status_code=500
        )


class ConfigurationError(APIError):
    """Raised when configuration errors occur."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details or {},
            status_code=400
        )


class SystemError(APIError):
    """Raised when system-level errors occur."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SYSTEM_ERROR",
            details=details or {},
            status_code=500
        )


def api_error_handler(func):
    """
    Decorator for API error handling.
    
    Catches exceptions and converts them to standardized error responses.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            logger.error(f"API Error in {func.__name__}: {e.message}", exc_info=True)
            return e.to_response()
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            error = APIError(
                message="Internal server error",
                error_code="INTERNAL_ERROR",
                details={"original_error": str(e)}
            )
            return error.to_response()
    return wrapper


def validate_request(request_data: Dict[str, Any], required_fields: List[str], 
                    optional_fields: List[str] = None) -> None:
    """
    Validate API request data.
    
    Args:
        request_data: The request data to validate
        required_fields: List of required field names
        optional_fields: List of optional field names
        
    Raises:
        ValidationError: If validation fails
    """
    if optional_fields is None:
        optional_fields = []
    
    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in request_data]
    if missing_fields:
        raise ValidationError(
            message=f"Missing required fields: {missing_fields}",
            details={"missing_fields": missing_fields}
        )
    
    # Check for unexpected fields
    allowed_fields = set(required_fields + optional_fields)
    unexpected_fields = [field for field in request_data.keys() if field not in allowed_fields]
    if unexpected_fields:
        raise ValidationError(
            message=f"Unexpected fields: {unexpected_fields}",
            details={"unexpected_fields": unexpected_fields}
        )


def validate_model_name(model_name: str) -> None:
    """
    Validate model name format.
    
    Args:
        model_name: The model name to validate
        
    Raises:
        ValidationError: If model name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValidationError("Model name must be a non-empty string", field="model_name")
    
    if len(model_name.strip()) == 0:
        raise ValidationError("Model name cannot be empty or whitespace", field="model_name")
    
    # Additional validation rules can be added here
    if len(model_name) > 100:
        raise ValidationError("Model name too long (max 100 characters)", field="model_name")


def validate_generation_config(config: Dict[str, Any]) -> None:
    """
    Validate generation configuration parameters.
    
    Args:
        config: Generation configuration dictionary
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Generation config must be a dictionary", field="generation_config")
    
    # Validate temperature
    if "temperature" in config:
        temp = config["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
            raise ValidationError(
                "Temperature must be a number between 0.0 and 2.0",
                field="generation_config.temperature"
            )
    
    # Validate max_new_tokens
    if "max_new_tokens" in config:
        tokens = config["max_new_tokens"]
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValidationError(
                "max_new_tokens must be a positive integer",
                field="generation_config.max_new_tokens"
            )
    
    # Validate top_p
    if "top_p" in config:
        top_p = config["top_p"]
        if not isinstance(top_p, (int, float)) or top_p < 0.0 or top_p > 1.0:
            raise ValidationError(
                "top_p must be a number between 0.0 and 1.0",
                field="generation_config.top_p"
            )


def validate_session_id(session_id: str) -> None:
    """
    Validate session ID format.
    
    Args:
        session_id: The session ID to validate
        
    Raises:
        ValidationError: If session ID is invalid
    """
    if not session_id or not isinstance(session_id, str):
        raise ValidationError("Session ID must be a non-empty string", field="session_id")
    
    if len(session_id.strip()) == 0:
        raise ValidationError("Session ID cannot be empty or whitespace", field="session_id")
    
    # Additional validation can be added here (UUID format, etc.)


class ErrorCodes:
    """Standard error codes for API responses."""
    
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    
    # Authentication/Authorization
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # Model errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_ALREADY_LOADED = "MODEL_ALREADY_LOADED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    
    # Configuration errors
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INVALID_CONFIG_VALUE = "INVALID_CONFIG_VALUE"
    CONFIG_FILE_ERROR = "CONFIG_FILE_ERROR"
    
    # System errors
    SYSTEM_ERROR = "SYSTEM_ERROR"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # Session errors
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_SAVE_ERROR = "SESSION_SAVE_ERROR"
    SESSION_LOAD_ERROR = "SESSION_LOAD_ERROR"
