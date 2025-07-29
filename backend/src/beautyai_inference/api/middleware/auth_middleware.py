"""
Authentication Middleware for BeautyAI Inference Framework.

Provides FastAPI middleware for authentication, authorization, and security.
"""

import time
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from ..auth import (
    validate_api_key,
    validate_jwt_token,
    get_user_permissions,
    get_user_roles,
    AuthConfig
)
from ..errors import AuthenticationError, AuthorizationError


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API requests.
    
    Handles JWT token validation, API key authentication, and permission checks.
    """
    
    def __init__(
        self,
        app,
        auth_config: Optional[AuthConfig] = None,
        skip_paths: Optional[list] = None
    ):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application instance
            auth_config: Authentication configuration
            skip_paths: List of paths to skip authentication
        """
        super().__init__(app)
        self.auth_config = auth_config or AuthConfig()
        self.skip_paths = skip_paths or [
            "/",
            "/health",
            "/health/basic",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process authentication for incoming requests.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        # Skip authentication for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Skip authentication in development mode
        if self.auth_config.development_mode:
            request.state.user = {
                "id": "dev_user",
                "username": "developer",
                "permissions": ["*"],
                "roles": ["admin"],
                "authenticated": True
            }
            return await call_next(request)
        
        try:
            # Extract authentication credentials
            auth_result = await self._authenticate_request(request)
            
            if not auth_result:
                raise AuthenticationError("Authentication required")
            
            # Store user context in request state
            request.state.user = auth_result
            
            # Continue to next middleware
            response = await call_next(request)
            
            return response
            
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except AuthorizationError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Authentication error")
    
    async def _authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Authenticate the incoming request.
        
        Args:
            request: HTTP request to authenticate
            
        Returns:
            User context if authenticated, None otherwise
        """
        # Try API key authentication first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._authenticate_api_key(api_key)
        
        # Try JWT token authentication
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            return await self._authenticate_jwt_token(token)
        
        return None
    
    async def _authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            User context if valid, None otherwise
        """
        try:
            # Validate API key (placeholder for future implementation)
            is_valid = await validate_api_key(api_key)
            
            if not is_valid:
                return None
            
            # Get user information based on API key
            # This would typically involve database lookup
            return {
                "id": f"api_key_user_{api_key[:8]}",
                "username": "api_user",
                "permissions": await get_user_permissions(api_key),
                "roles": await get_user_roles(api_key),
                "authenticated": True,
                "auth_method": "api_key"
            }
            
        except Exception:
            return None
    
    async def _authenticate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate using JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            User context if valid, None otherwise
        """
        try:
            # Validate JWT token (placeholder for future implementation)
            payload = await validate_jwt_token(token)
            
            if not payload:
                return None
            
            return {
                "id": payload.get("user_id"),
                "username": payload.get("username"),
                "permissions": payload.get("permissions", []),
                "roles": payload.get("roles", []),
                "authenticated": True,
                "auth_method": "jwt",
                "token_exp": payload.get("exp")
            }
            
        except Exception:
            return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for API requests.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application instance
            requests_per_minute: Maximum requests per minute per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Apply rate limiting to incoming requests.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.request_counts[client_ip] = []
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        # Record this request
        self.request_counts[client_ip].append(current_time)
        
        # Continue to next middleware
        response = await call_next(request)
        
        return response
