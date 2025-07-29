"""
Base API Service Adapter.

Provides the foundational adapter class that bridges BeautyAI services
with API endpoints, handling common patterns like request validation,
response formatting, and error handling.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
import logging
from datetime import datetime
import time

from ..models import APIRequest, APIResponse, ErrorResponse
from ..errors import APIError, ValidationError, api_error_handler
from ..auth import AuthContext, require_auth
from ...services.base.base_service import BaseService

logger = logging.getLogger(__name__)


class APIServiceAdapter(ABC):
    """
    Base adapter class for bridging services with API endpoints.
    
    This class provides common functionality for request/response handling,
    validation, authentication, and error management that all API adapters need.
    """
    
    def __init__(self, service: BaseService):
        """
        Initialize the adapter with a service instance.
        
        Args:
            service: The BeautyAI service instance to adapt for API use
        """
        self.service = service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_request(self, request: APIRequest, 
                        request_type: Type[APIRequest]) -> None:
        """
        Validate incoming API request.
        
        Args:
            request: The request object to validate
            request_type: Expected request type class
            
        Raises:
            ValidationError: If request validation fails
        """
        if not isinstance(request, request_type):
            raise ValidationError(
                f"Invalid request type. Expected {request_type.__name__}, "
                f"got {type(request).__name__}"
            )
    
    def create_success_response(self, data: Dict[str, Any], 
                               request_start_time: float) -> APIResponse:
        """
        Create a successful API response.
        
        Args:
            data: Response data dictionary
            request_start_time: Request start timestamp for execution time calculation
            
        Returns:
            APIResponse with success status and execution time
        """
        execution_time = (time.time() - request_start_time) * 1000  # Convert to ms
        
        return APIResponse(
            success=True,
            data=data,
            execution_time_ms=round(execution_time, 2)
        )
    
    def create_error_response(self, error: Exception, 
                             request_start_time: float) -> ErrorResponse:
        """
        Create an error API response from an exception.
        
        Args:
            error: The exception that occurred
            request_start_time: Request start timestamp for execution time calculation
            
        Returns:
            ErrorResponse with error details and execution time
        """
        execution_time = (time.time() - request_start_time) * 1000  # Convert to ms
        
        if isinstance(error, APIError):
            response = error.to_response()
            response.execution_time_ms = round(execution_time, 2)
            return response
        
        # Handle unexpected errors
        self.logger.error(f"Unexpected error in API adapter: {error}", exc_info=True)
        return ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            error_message="An unexpected error occurred",
            error_details={"error_type": type(error).__name__},
            execution_time_ms=round(execution_time, 2)
        )
    
    @api_error_handler
    def execute_with_auth(self, auth_context: AuthContext, 
                         operation: callable, *args, **kwargs) -> APIResponse:
        """
        Execute an operation with authentication and error handling.
        
        Args:
            auth_context: Authentication context for the request
            operation: The operation to execute
            *args, **kwargs: Arguments to pass to the operation
            
        Returns:
            APIResponse with operation result or error details
        """
        request_start_time = time.time()
        
        try:
            # Execute the operation
            result = operation(*args, **kwargs)
            
            # Convert result to API response format
            if isinstance(result, dict):
                return self.create_success_response(result, request_start_time)
            elif hasattr(result, 'to_dict'):
                return self.create_success_response(result.to_dict(), request_start_time)
            else:
                return self.create_success_response({"result": result}, request_start_time)
                
        except Exception as e:
            return self.create_error_response(e, request_start_time)
    
    @abstractmethod
    def get_supported_operations(self) -> Dict[str, str]:
        """
        Get the list of operations supported by this adapter.
        
        Returns:
            Dictionary mapping operation names to their descriptions
        """
        pass
