"""
Error Middleware for BeautyAI Inference Framework.

Provides FastAPI middleware for centralized error handling and response formatting.
"""

import traceback
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..errors import (
    BeautyAIError,
    ModelError,
    InferenceError,
    ConfigurationError,
    AuthenticationError,
    AuthorizationError,
    ValidationError
)


class ErrorMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware for API requests.
    
    Provides centralized error handling, logging, and response formatting.
    """
    
    def __init__(self, app, debug: bool = False):
        """
        Initialize error middleware.
        
        Args:
            app: FastAPI application instance
            debug: Whether to include debug information in error responses
        """
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle errors from API requests with proper formatting.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response with error handling
        """
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # FastAPI HTTP exceptions - pass through
            return await self._create_error_response(
                request=request,
                status_code=e.status_code,
                error_type="HTTPException",
                message=e.detail,
                details=getattr(e, 'headers', None)
            )
            
        except BeautyAIError as e:
            # Custom BeautyAI framework errors
            return await self._handle_beautyai_error(request, e)
            
        except ValueError as e:
            # Validation errors
            return await self._create_error_response(
                request=request,
                status_code=400,
                error_type="ValidationError",
                message=str(e)
            )
            
        except PermissionError as e:
            # Permission errors
            return await self._create_error_response(
                request=request,
                status_code=403,
                error_type="PermissionError",
                message=str(e)
            )
            
        except FileNotFoundError as e:
            # File not found errors
            return await self._create_error_response(
                request=request,
                status_code=404,
                error_type="FileNotFoundError",
                message=str(e)
            )
            
        except Exception as e:
            # Unexpected errors
            return await self._handle_unexpected_error(request, e)
    
    async def _handle_beautyai_error(self, request: Request, error: BeautyAIError) -> JSONResponse:
        """
        Handle BeautyAI framework specific errors.
        
        Args:
            request: HTTP request
            error: BeautyAI error instance
            
        Returns:
            JSON error response
        """
        # Map error types to HTTP status codes
        status_code_map = {
            ModelError: 422,
            InferenceError: 500,
            ConfigurationError: 400,
            AuthenticationError: 401,
            AuthorizationError: 403,
            ValidationError: 400,
            BeautyAIError: 500  # Generic framework error
        }
        
        status_code = status_code_map.get(type(error), 500)
        
        return await self._create_error_response(
            request=request,
            status_code=status_code,
            error_type=error.__class__.__name__,
            message=str(error),
            details=getattr(error, 'details', None)
        )
    
    async def _handle_unexpected_error(self, request: Request, error: Exception) -> JSONResponse:
        """
        Handle unexpected errors with proper logging.
        
        Args:
            request: HTTP request
            error: Exception instance
            
        Returns:
            JSON error response
        """
        # Log the full traceback for debugging
        error_details = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc() if self.debug else None
        }
        
        # In production, log this to your logging system
        # logger.error("Unexpected API error", extra=error_details)
        
        return await self._create_error_response(
            request=request,
            status_code=500,
            error_type="InternalServerError",
            message="An unexpected error occurred",
            details=error_details if self.debug else None
        )
    
    async def _create_error_response(
        self,
        request: Request,
        status_code: int,
        error_type: str,
        message: str,
        details: dict = None
    ) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            request: HTTP request
            status_code: HTTP status code
            error_type: Type of error
            message: Error message
            details: Additional error details
            
        Returns:
            JSON error response
        """
        error_response = {
            "error": {
                "type": error_type,
                "message": message,
                "status_code": status_code,
                "timestamp": self._get_timestamp(),
                "request_id": getattr(request.state, 'request_id', None),
                "path": request.url.path,
                "method": request.method
            }
        }
        
        if details:
            error_response["error"]["details"] = details
        
        if self.debug:
            error_response["error"]["debug"] = {
                "user_agent": request.headers.get("User-Agent"),
                "client_ip": request.client.host if request.client else None,
                "query_params": dict(request.query_params)
            }
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            ISO formatted timestamp
        """
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


class ValidationErrorMiddleware(BaseHTTPMiddleware):
    """
    Validation error middleware for Pydantic validation errors.
    """
    
    def __init__(self, app):
        """
        Initialize validation error middleware.
        
        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle Pydantic validation errors.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response with validation error handling
        """
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            # Check if it's a Pydantic validation error
            if hasattr(e, 'errors') and callable(getattr(e, 'errors')):
                return await self._handle_validation_error(request, e)
            
            # Re-raise if not a validation error
            raise e
    
    async def _handle_validation_error(self, request: Request, error) -> JSONResponse:
        """
        Handle Pydantic validation errors with detailed field information.
        
        Args:
            request: HTTP request
            error: Pydantic validation error
            
        Returns:
            JSON error response with field details
        """
        validation_errors = []
        
        for err in error.errors():
            validation_errors.append({
                "field": ".".join(str(x) for x in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
                "input": err.get("input")
            })
        
        error_response = {
            "error": {
                "type": "ValidationError",
                "message": "Request validation failed",
                "status_code": 422,
                "timestamp": self._get_timestamp(),
                "request_id": getattr(request.state, 'request_id', None),
                "validation_errors": validation_errors
            }
        }
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            ISO formatted timestamp
        """
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
