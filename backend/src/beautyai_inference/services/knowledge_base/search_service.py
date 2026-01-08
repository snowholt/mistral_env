"""
Search service for knowledge base RAG.

Provides vector similarity search using pgvector and reranking.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from ...database.models import Chunk, Document, KnowledgeBase, DocumentStatus
from .embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the knowledge base."""
    chunk_id: int
    document_id: int
    document_title: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchService:
    """
    Vector search service for knowledge base.
    
    Uses pgvector for similarity search with optional reranking.
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """Initialize with optional embedding service."""
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Search settings
        self.default_top_k = int(os.getenv("KB_SEARCH_TOP_K", "5"))
        self.similarity_threshold = float(os.getenv("KB_SIMILARITY_THRESHOLD", "0.5"))
    
    async def search(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search the knowledge base for relevant chunks.
        
        Uses cosine similarity search with pgvector.
        
        Args:
            db: Database session
            knowledge_base_id: Knowledge base to search
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of SearchResult objects ordered by relevance
        """
        top_k = top_k or self.default_top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Build the vector search query using pgvector
        # Using cosine distance: 1 - (a <=> b) gives similarity
        # The <=> operator is cosine distance, so we use 1 - distance for similarity
        sql = text("""
            SELECT 
                c.id as chunk_id,
                c.document_id,
                d.title as document_title,
                c.content,
                c.metadata as chunk_metadata,
                1 - (c.embedding <=> :query_embedding::vector) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.knowledge_base_id = :kb_id
              AND d.status = :ready_status
              AND 1 - (c.embedding <=> :query_embedding::vector) >= :threshold
            ORDER BY c.embedding <=> :query_embedding::vector
            LIMIT :limit
        """)
        
        result = await db.execute(
            sql,
            {
                "kb_id": knowledge_base_id,
                "query_embedding": str(query_embedding),
                "ready_status": DocumentStatus.READY.value,
                "threshold": similarity_threshold,
                "limit": top_k,
            }
        )
        
        rows = result.fetchall()
        
        results = [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                document_title=row.document_title,
                content=row.content,
                score=float(row.similarity),
                metadata=row.chunk_metadata or {},
            )
            for row in rows
        ]
        
        logger.info(f"Search returned {len(results)} results for KB {knowledge_base_id}")
        return results
    
    async def search_multi_kb(
        self,
        db: AsyncSession,
        knowledge_base_ids: List[int],
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search across multiple knowledge bases.
        
        Useful for customers with multiple knowledge bases.
        """
        if not knowledge_base_ids:
            return []
        
        top_k = top_k or self.default_top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Build query for multiple knowledge bases
        sql = text("""
            SELECT 
                c.id as chunk_id,
                c.document_id,
                d.title as document_title,
                c.content,
                c.metadata as chunk_metadata,
                1 - (c.embedding <=> :query_embedding::vector) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.knowledge_base_id = ANY(:kb_ids)
              AND d.status = :ready_status
              AND 1 - (c.embedding <=> :query_embedding::vector) >= :threshold
            ORDER BY c.embedding <=> :query_embedding::vector
            LIMIT :limit
        """)
        
        result = await db.execute(
            sql,
            {
                "kb_ids": knowledge_base_ids,
                "query_embedding": str(query_embedding),
                "ready_status": DocumentStatus.READY.value,
                "threshold": similarity_threshold,
                "limit": top_k,
            }
        )
        
        rows = result.fetchall()
        
        return [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                document_title=row.document_title,
                content=row.content,
                score=float(row.similarity),
                metadata=row.chunk_metadata or {},
            )
            for row in rows
        ]
    
    async def get_context_for_rag(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        query: str,
        max_tokens: int = 2000,
        top_k: int = 5,
    ) -> str:
        """
        Get formatted context for RAG prompt injection.
        
        Returns a string of relevant document chunks formatted
        for use in an LLM prompt.
        """
        results = await self.search(
            db=db,
            knowledge_base_id=knowledge_base_id,
            query=query,
            top_k=top_k,
        )
        
        if not results:
            return ""
        
        # Format context with source attribution
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate of chars per token
        
        for i, result in enumerate(results, 1):
            chunk_text = f"[Source: {result.document_title}]\n{result.content}"
            
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n\n---\n\n".join(context_parts)
    
    async def get_knowledge_base_stats(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
    ) -> Dict[str, Any]:
        """Get statistics about a knowledge base."""
        # Count documents
        doc_count_result = await db.execute(
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
        )
        documents = doc_count_result.scalars().all()
        
        # Count by status
        status_counts = {}
        total_chunks = 0
        for doc in documents:
            status = doc.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_chunks += doc.chunk_count or 0
        
        # Get knowledge base info
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
        )
        kb = kb_result.scalar_one_or_none()
        
        return {
            "knowledge_base_id": knowledge_base_id,
            "name": kb.name if kb else None,
            "document_count": len(documents),
            "chunk_count": total_chunks,
            "status_breakdown": status_counts,
            "ready_documents": status_counts.get(DocumentStatus.READY.value, 0),
        }


# Singleton instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create singleton search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
