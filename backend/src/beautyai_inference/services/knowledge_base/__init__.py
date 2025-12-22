"""
Knowledge Base services for RAG (Retrieval-Augmented Generation).

This module provides document processing, embedding generation,
and vector search capabilities using pgvector.
"""

from .embedding_service import EmbeddingService, get_embedding_service
from .document_processor import DocumentProcessor, get_document_processor
from .search_service import SearchService, get_search_service, SearchResult

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "DocumentProcessor",
    "get_document_processor",
    "SearchService",
    "get_search_service",
    "SearchResult",
]
