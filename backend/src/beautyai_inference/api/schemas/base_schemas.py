"""
Base API Schemas.

Defines base classes for API requests and responses,
providing common functionality and structure.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from abc import ABC


@dataclass
class BaseRequest(ABC):
    """Base class for all API request schemas."""
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if getattr(self, field.name) is not None
        }


@dataclass
class BaseResponse(ABC):
    """Base class for all API response schemas."""
    success: bool = True
    message: Optional[str] = None
    error_code: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if getattr(self, field.name) is not None
        }


@dataclass
class ErrorResponse(BaseResponse):
    """Standard error response schema."""
    success: bool = False
    error_details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None


@dataclass
class PaginatedRequest(BaseRequest):
    """Base class for paginated requests."""
    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # "asc" or "desc"


@dataclass
class PaginatedResponse(BaseResponse):
    """Base class for paginated responses."""
    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool
