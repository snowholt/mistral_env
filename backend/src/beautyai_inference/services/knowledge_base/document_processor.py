"""
Document processor for knowledge base.

Handles document upload, text extraction, chunking, and embedding generation.
Supports various document formats: PDF, TXT, DOCX, Markdown.
"""

import os
import re
import logging
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...database.models import Document, Chunk, DocumentStatus, KnowledgeBase
from .embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents for the knowledge base.
    
    Workflow:
    1. Upload document → extract text
    2. Chunk text into smaller segments
    3. Generate embeddings for each chunk
    4. Store chunks with embeddings in PostgreSQL/pgvector
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """Initialize with optional embedding service."""
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Chunking settings
        self.chunk_size = int(os.getenv("KB_CHUNK_SIZE", "1000"))  # characters
        self.chunk_overlap = int(os.getenv("KB_CHUNK_OVERLAP", "200"))  # characters
        self.min_chunk_size = int(os.getenv("KB_MIN_CHUNK_SIZE", "100"))  # characters
    
    async def process_document(
        self,
        db: AsyncSession,
        document_id: int,
        file_path: str,
        file_type: str,
    ) -> Tuple[bool, str]:
        """
        Process a document: extract text, chunk, embed, and store.
        
        Args:
            db: Database session
            document_id: Document ID in database
            file_path: Path to the uploaded file
            file_type: MIME type of the file
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Update document status to processing
            await self._update_document_status(db, document_id, DocumentStatus.PROCESSING)
            
            # Extract text based on file type
            text = await self._extract_text(file_path, file_type)
            if not text or not text.strip():
                await self._update_document_status(
                    db, document_id, DocumentStatus.FAILED, "No text extracted from document"
                )
                return False, "No text extracted from document"
            
            # Chunk the text
            chunks = self._chunk_text(text)
            if not chunks:
                await self._update_document_status(
                    db, document_id, DocumentStatus.FAILED, "No chunks created"
                )
                return False, "No chunks created from document"
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = await self.embedding_service.embed_texts(chunks)
            
            # Store chunks with embeddings
            await self._store_chunks(db, document_id, chunks, embeddings)
            
            # Update document status
            await self._update_document_status(
                db, document_id, DocumentStatus.READY,
                chunk_count=len(chunks),
                extracted_text=text[:10000],  # Store first 10k chars for reference
            )
            
            logger.info(f"Document {document_id} processed: {len(chunks)} chunks")
            return True, f"Processed {len(chunks)} chunks"
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            await self._update_document_status(
                db, document_id, DocumentStatus.FAILED, str(e)
            )
            return False, str(e)
    
    async def _extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from various document formats."""
        file_type = file_type.lower()
        
        if file_type in ["text/plain", ".txt"]:
            return self._extract_txt(file_path)
        
        elif file_type in ["text/markdown", ".md"]:
            return self._extract_markdown(file_path)
        
        elif file_type in ["application/pdf", ".pdf"]:
            return await self._extract_pdf(file_path)
        
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".docx"
        ]:
            return await self._extract_docx(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _extract_markdown(self, file_path: str) -> str:
        """Extract text from markdown file (remove formatting)."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Remove common markdown formatting for cleaner text
        # Keep the text content, remove syntax
        content = re.sub(r"```[\s\S]*?```", "", content)  # Code blocks
        content = re.sub(r"`[^`]+`", "", content)  # Inline code
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)  # Images
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)  # Links → text
        content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)  # Headers
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)  # Bold
        content = re.sub(r"\*([^*]+)\*", r"\1", content)  # Italic
        content = re.sub(r"^[-*+]\s+", "", content, flags=re.MULTILINE)  # List items
        
        return content.strip()
    
    async def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import pymupdf  # PyMuPDF (fitz)
            
            text_parts = []
            with pymupdf.open(file_path) as doc:
                for page in doc:
                    text_parts.append(page.get_text())
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.error("pymupdf not installed. Run: pip install pymupdf")
            raise
    
    async def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document as DocxDocument
            
            doc = DocxDocument(file_path)
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Uses sentence-aware splitting to avoid cutting mid-sentence.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split into sentences (rough, but works for most cases)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap from end of current
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + " " + sentence
                else:
                    # Chunk too small, keep accumulating
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Don't forget the last chunk
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _store_chunks(
        self,
        db: AsyncSession,
        document_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """Store chunks with their embeddings in the database."""
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = Chunk(
                document_id=document_id,
                content=chunk_text,
                chunk_index=idx,
                embedding=embedding,
                metadata={
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                },
            )
            db.add(chunk)
        
        await db.commit()
    
    async def _update_document_status(
        self,
        db: AsyncSession,
        document_id: int,
        status: DocumentStatus,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        extracted_text: Optional[str] = None,
    ) -> None:
        """Update document status in database."""
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if document:
            document.status = status
            if error_message:
                document.error_message = error_message
            if chunk_count is not None:
                document.chunk_count = chunk_count
            if extracted_text:
                document.extracted_text = extracted_text
            document.processed_at = datetime.now(timezone.utc)
            await db.commit()


# Singleton instance
_document_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get or create singleton document processor instance."""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
