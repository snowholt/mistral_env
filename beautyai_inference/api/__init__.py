"""
API Foundation Package for BeautyAI.

This package provides the foundation for future REST/GraphQL API integration:
- Request/response data models
- Authentication and authorization hooks
- API-compatible error handling
- Service adapter patterns for API endpoints
"""

from .models import *
from .auth import *
from .errors import *
from .adapters import *

__all__ = [
    # Request/Response models
    'APIRequest', 'APIResponse', 'ErrorResponse',
    'ChatRequest', 'ChatResponse', 'ModelRequest', 'ModelResponse',
    'ConfigRequest', 'ConfigResponse', 'SystemRequest', 'SystemResponse',
    
    # Authentication
    'AuthContext', 'authenticate', 'authorize',
    
    # Error handling
    'APIError', 'ValidationError', 'AuthenticationError', 'AuthorizationError',
    'api_error_handler', 'validate_request',
    
    # Service adapters
    'APIServiceAdapter', 'ChatAPIAdapter', 'ModelAPIAdapter',
    'ConfigAPIAdapter', 'SystemAPIAdapter'
]

__version__ = "1.0.0"
