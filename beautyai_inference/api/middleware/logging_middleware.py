"""
Logging Middleware for BeautyAI Inference Framework.

Provides FastAPI middleware for request/response logging and monitoring.
"""

import json
import time
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.memory_utils import get_memory_info


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for API requests and responses.
    
    Provides comprehensive logging of API interactions for monitoring and debugging.
    """
    
    def __init__(
        self,
        app,
        logger: Optional[logging.Logger] = None,
        log_requests: bool = True,
        log_responses: bool = True,
        log_body: bool = False,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application instance
            logger: Logger instance to use
            log_requests: Whether to log request details
            log_responses: Whether to log response details
            log_body: Whether to log request/response bodies
            exclude_paths: Paths to exclude from logging
        """
        super().__init__(app)
        self.logger = logger or logging.getLogger("beautyai.api")
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_body = log_body
        self.exclude_paths = exclude_paths or [
            "/health/basic",
            "/metrics",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response details.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Log request if enabled
        if self.log_requests:
            await self._log_request(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log response if enabled
        if self.log_responses:
            await self._log_response(request, response, processing_time)
        
        return response
    
    async def _log_request(self, request: Request) -> None:
        """
        Log incoming request details.
        
        Args:
            request: HTTP request to log
        """
        # Get basic request information
        request_data = {
            "request_id": getattr(request.state, 'request_id', None),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("User-Agent"),
            "content_type": request.headers.get("Content-Type"),
            "content_length": request.headers.get("Content-Length"),
            "timestamp": time.time()
        }
        
        # Add user information if available
        if hasattr(request.state, 'user'):
            request_data["user"] = {
                "id": request.state.user.get("id"),
                "username": request.state.user.get("username"),
                "auth_method": request.state.user.get("auth_method")
            }
        
        # Add body if logging is enabled and safe to do so
        if self.log_body and self._should_log_body(request):
            try:
                body = await self._get_request_body(request)
                if body:
                    request_data["body"] = body
            except Exception as e:
                request_data["body_error"] = str(e)
        
        # Add memory information
        memory_info = get_memory_info()
        request_data["memory_usage_mb"] = memory_info.get("used_mb", 0)
        
        # Log request
        self.logger.info(
            f"API Request: {request.method} {request.url.path}",
            extra={"request": request_data}
        )
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        processing_time: float
    ) -> None:
        """
        Log response details.
        
        Args:
            request: HTTP request
            response: HTTP response
            processing_time: Time taken to process request
        """
        # Get response information
        response_data = {
            "request_id": getattr(request.state, 'request_id', None),
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        
        # Add body if logging is enabled and safe to do so
        if self.log_body and self._should_log_response_body(response):
            try:
                body = await self._get_response_body(response)
                if body:
                    response_data["body"] = body
            except Exception as e:
                response_data["body_error"] = str(e)
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
        elif response.status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Log response
        self.logger.log(
            log_level,
            f"API Response: {response.status_code} ({processing_time:.3f}s)",
            extra={"response": response_data}
        )
    
    async def _get_request_body(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Safely get request body for logging.
        
        Args:
            request: HTTP request
            
        Returns:
            Request body as dictionary or None
        """
        try:
            content_type = request.headers.get("Content-Type", "")
            
            if content_type.startswith("application/json"):
                body = await request.body()
                if body:
                    return json.loads(body.decode())
            
            return None
            
        except Exception:
            return None
    
    async def _get_response_body(self, response: Response) -> Optional[Dict[str, Any]]:
        """
        Safely get response body for logging.
        
        Args:
            response: HTTP response
            
        Returns:
            Response body as dictionary or None
        """
        try:
            # This is tricky with FastAPI responses
            # In practice, you might want to use a custom response class
            # or capture the body before it's sent
            return None
            
        except Exception:
            return None
    
    def _should_log_body(self, request: Request) -> bool:
        """
        Determine if request body should be logged.
        
        Args:
            request: HTTP request
            
        Returns:
            True if body should be logged
        """
        # Don't log bodies for certain content types
        content_type = request.headers.get("Content-Type", "")
        
        if content_type.startswith("multipart/form-data"):
            return False
        
        if content_type.startswith("application/octet-stream"):
            return False
        
        # Don't log very large bodies
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > 10240:  # 10KB
            return False
        
        return True
    
    def _should_log_response_body(self, response: Response) -> bool:
        """
        Determine if response body should be logged.
        
        Args:
            response: HTTP response
            
        Returns:
            True if body should be logged
        """
        # Don't log large responses
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > 10240:  # 10KB
            return False
        
        # Don't log binary content
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith(("application/json", "text/")):
            return False
        
        return True


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Performance logging middleware for monitoring API performance.
    """
    
    def __init__(
        self,
        app,
        logger: Optional[logging.Logger] = None,
        slow_request_threshold: float = 1.0
    ):
        """
        Initialize performance logging middleware.
        
        Args:
            app: FastAPI application instance
            logger: Logger instance to use
            slow_request_threshold: Threshold in seconds for slow request logging
        """
        super().__init__(app)
        self.logger = logger or logging.getLogger("beautyai.performance")
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Monitor and log performance metrics.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        # Record start metrics
        start_time = time.time()
        start_memory = get_memory_info()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        end_memory = get_memory_info()
        memory_delta = end_memory.get("used_mb", 0) - start_memory.get("used_mb", 0)
        
        # Log performance metrics
        performance_data = {
            "request_id": getattr(request.state, 'request_id', None),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "memory_delta_mb": memory_delta,
            "start_memory_mb": start_memory.get("used_mb", 0),
            "end_memory_mb": end_memory.get("used_mb", 0),
            "timestamp": time.time()
        }
        
        # Log slow requests with warning level
        if processing_time > self.slow_request_threshold:
            self.logger.warning(
                f"Slow API Request: {request.method} {request.url.path} ({processing_time:.3f}s)",
                extra={"performance": performance_data}
            )
        else:
            self.logger.info(
                f"API Performance: {request.method} {request.url.path}",
                extra={"performance": performance_data}
            )
        
        return response
