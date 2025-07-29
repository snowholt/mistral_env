"""
Request Middleware for BeautyAI Inference Framework.

Provides FastAPI middleware for request processing, validation, and enrichment.
"""

import time
import uuid
from typing import Callable, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..models import RequestMetadata
from ...utils.memory_utils import get_memory_info


class RequestMiddleware(BaseHTTPMiddleware):
    """
    Request processing middleware for API requests.
    
    Handles request ID generation, timing, metadata collection, and enrichment.
    """
    
    def __init__(self, app, collect_metrics: bool = True):
        """
        Initialize request middleware.
        
        Args:
            app: FastAPI application instance
            collect_metrics: Whether to collect request metrics
        """
        super().__init__(app)
        self.collect_metrics = collect_metrics
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process incoming requests with metadata and timing.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response with added headers
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Record start time
        start_time = time.time()
        
        # Collect request metadata
        metadata = await self._collect_request_metadata(request, request_id)
        
        # Store metadata in request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        request.state.metadata = metadata
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        
        # Collect metrics if enabled
        if self.collect_metrics:
            await self._record_metrics(request, response, processing_time)
        
        return response
    
    async def _collect_request_metadata(self, request: Request, request_id: str) -> RequestMetadata:
        """
        Collect metadata about the incoming request.
        
        Args:
            request: HTTP request
            request_id: Unique request identifier
            
        Returns:
            Request metadata object
        """
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Get memory information
        memory_info = get_memory_info()
        
        return RequestMetadata(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
            content_type=request.headers.get("Content-Type"),
            content_length=request.headers.get("Content-Length"),
            timestamp=time.time(),
            memory_usage_mb=memory_info.get("used_mb", 0)
        )
    
    async def _record_metrics(
        self,
        request: Request,
        response: Response,
        processing_time: float
    ) -> None:
        """
        Record request metrics for monitoring.
        
        Args:
            request: HTTP request
            response: HTTP response
            processing_time: Request processing time in seconds
        """
        # This would typically send metrics to a monitoring system
        # For now, we'll just log them (placeholder for future implementation)
        metrics = {
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        
        # In a real implementation, this would send to metrics collection service
        # logger.info("Request metrics", extra=metrics)


class ContentValidationMiddleware(BaseHTTPMiddleware):
    """
    Content validation middleware for API requests.
    """
    
    def __init__(self, app, max_content_length: int = 10 * 1024 * 1024):  # 10MB default
        """
        Initialize content validation middleware.
        
        Args:
            app: FastAPI application instance
            max_content_length: Maximum allowed content length in bytes
        """
        super().__init__(app)
        self.max_content_length = max_content_length
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Validate request content before processing.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        # Check content length
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_content_length:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=413,
                detail=f"Content too large. Maximum allowed: {self.max_content_length} bytes"
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("Content-Type", "")
            if not content_type.startswith(("application/json", "multipart/form-data")):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported content type. Expected application/json or multipart/form-data"
                )
        
        # Continue to next middleware
        response = await call_next(request)
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware for API requests.
    """
    
    def __init__(
        self,
        app,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None
    ):
        """
        Initialize CORS middleware.
        
        Args:
            app: FastAPI application instance
            allow_origins: Allowed origins for CORS
            allow_methods: Allowed HTTP methods
            allow_headers: Allowed headers
        """
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or [
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle CORS for incoming requests.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response with CORS headers
        """
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*" if "*" in self.allow_origins else request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        response.headers["Access-Control-Max-Age"] = "86400"
        
        return response
