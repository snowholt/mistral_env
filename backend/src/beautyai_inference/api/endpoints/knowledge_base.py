"""
Knowledge Base API endpoints for RAG functionality.

Endpoints:
- POST /kb - Create knowledge base
- GET /kb - List knowledge bases
- GET /kb/{id} - Get knowledge base details
- DELETE /kb/{id} - Delete knowledge base
- POST /kb/{id}/documents - Upload document
- GET /kb/{id}/documents - List documents
- DELETE /kb/{id}/documents/{doc_id} - Delete document
- POST /kb/{id}/search - Search knowledge base
"""

import os
import logging
import uuid
import shutil
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.connection import get_db
from ...database.models import (
    User, Customer, KnowledgeBase, Document, Chunk, DocumentStatus
)
from ..endpoints.whatsapp_auth import get_current_user
from ...services.knowledge_base import (
    get_document_processor, 
    get_search_service,
    SearchResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/kb", tags=["Knowledge Base"])

# Upload directory
UPLOAD_DIR = Path(os.getenv("KB_UPLOAD_DIR", "/tmp/beautyai_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file types
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateKnowledgeBaseRequest(BaseModel):
    """Request to create a knowledge base."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class KnowledgeBaseResponse(BaseModel):
    """Knowledge base details."""
    id: int
    name: str
    description: Optional[str]
    document_count: int
    chunk_count: int
    created_at: datetime
    updated_at: Optional[datetime]


class DocumentResponse(BaseModel):
    """Document details."""
    id: int
    title: str
    file_name: str
    file_type: str
    file_size: int
    status: str
    chunk_count: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]


class UploadResponse(BaseModel):
    """Response after document upload."""
    document_id: int
    title: str
    status: str
    message: str


class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchResultResponse(BaseModel):
    """Single search result."""
    chunk_id: int
    document_id: int
    document_title: str
    content: str
    score: float


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: List[SearchResultResponse]
    total_results: int


class StatsResponse(BaseModel):
    """Knowledge base statistics."""
    knowledge_base_id: int
    name: Optional[str]
    document_count: int
    chunk_count: int
    status_breakdown: dict
    ready_documents: int


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True


# ============================================================================
# Helper Functions
# ============================================================================

async def get_customer_for_user(db: AsyncSession, user: User) -> Customer:
    """Get customer associated with user.
    
    If user has multiple customers, returns the first one (by created_at).
    In the future, this should be changed to require customer_id parameter.
    """
    result = await db.execute(
        select(Customer)
        .where(Customer.user_id == user.id)
        .order_by(Customer.created_at.asc())
        .limit(1)
    )
    customer = result.scalar_one_or_none()
    if not customer:
        raise HTTPException(status_code=404, detail="No business profile found")
    return customer


async def get_kb_for_customer(
    db: AsyncSession, 
    kb_id: int, 
    customer_id: int
) -> KnowledgeBase:
    """Get knowledge base owned by customer."""
    result = await db.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.id == kb_id)
        .where(KnowledgeBase.customer_id == customer_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


# ============================================================================
# Knowledge Base CRUD
# ============================================================================

@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new knowledge base.
    
    Each customer can have multiple knowledge bases for organizing documents.
    """
    customer = await get_customer_for_user(db, current_user)
    
    # Create knowledge base
    kb = KnowledgeBase(
        customer_id=customer.id,
        name=request.name,
        description=request.description,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    
    logger.info(f"Knowledge base created: {kb.id} for customer {customer.id}")
    
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        document_count=0,
        chunk_count=0,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.get("", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all knowledge bases for the current user's business."""
    customer = await get_customer_for_user(db, current_user)
    
    result = await db.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.customer_id == customer.id)
        .order_by(KnowledgeBase.created_at.desc())
    )
    knowledge_bases = result.scalars().all()
    
    responses = []
    for kb in knowledge_bases:
        # Get document and chunk counts
        doc_result = await db.execute(
            select(Document).where(Document.knowledge_base_id == kb.id)
        )
        documents = doc_result.scalars().all()
        
        chunk_count = sum(d.chunk_count or 0 for d in documents)
        
        responses.append(KnowledgeBaseResponse(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            document_count=len(documents),
            chunk_count=chunk_count,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        ))
    
    return responses


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get knowledge base details."""
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    # Get counts
    doc_result = await db.execute(
        select(Document).where(Document.knowledge_base_id == kb.id)
    )
    documents = doc_result.scalars().all()
    chunk_count = sum(d.chunk_count or 0 for d in documents)
    
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        document_count=len(documents),
        chunk_count=chunk_count,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.delete("/{kb_id}", response_model=MessageResponse)
async def delete_knowledge_base(
    kb_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a knowledge base and all its documents.
    
    This action is irreversible!
    """
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    # Delete all chunks for documents in this KB
    await db.execute(
        delete(Chunk).where(
            Chunk.document_id.in_(
                select(Document.id).where(Document.knowledge_base_id == kb_id)
            )
        )
    )
    
    # Delete all documents
    await db.execute(
        delete(Document).where(Document.knowledge_base_id == kb_id)
    )
    
    # Delete knowledge base
    await db.delete(kb)
    await db.commit()
    
    logger.info(f"Knowledge base deleted: {kb_id}")
    
    return MessageResponse(message="Knowledge base deleted successfully")


# ============================================================================
# Document Management
# ============================================================================

@router.post("/{kb_id}/documents", response_model=UploadResponse, status_code=201)
async def upload_document(
    kb_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document to the knowledge base.
    
    Supported formats: PDF, TXT, Markdown, DOCX
    
    The document will be processed in the background:
    1. Text extraction
    2. Chunking
    3. Embedding generation
    """
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    # Validate file type
    content_type = file.content_type
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {list(ALLOWED_TYPES.keys())}"
        )
    
    # Generate unique filename
    ext = ALLOWED_TYPES[content_type]
    unique_name = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / unique_name
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Create document record
    doc_title = title or file.filename or "Untitled Document"
    
    document = Document(
        knowledge_base_id=kb.id,
        title=doc_title,
        file_name=file.filename,
        file_path=str(file_path),
        file_type=content_type,
        file_size=len(content),
        status=DocumentStatus.PENDING,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    
    # Process document in background
    # Note: In production, use a proper task queue like Celery
    processor = get_document_processor()
    background_tasks.add_task(
        _process_document_background,
        document.id,
        str(file_path),
        content_type,
    )
    
    logger.info(f"Document uploaded: {document.id} to KB {kb_id}")
    
    return UploadResponse(
        document_id=document.id,
        title=doc_title,
        status=DocumentStatus.PENDING.value,
        message="Document uploaded. Processing will begin shortly.",
    )


async def _process_document_background(
    document_id: int,
    file_path: str,
    file_type: str,
):
    """Background task to process document."""
    from ...database.connection import async_session_factory
    
    async with async_session_factory() as db:
        try:
            processor = get_document_processor()
            success, message = await processor.process_document(
                db=db,
                document_id=document_id,
                file_path=file_path,
                file_type=file_type,
            )
            logger.info(f"Document {document_id} processing: {success} - {message}")
        except Exception as e:
            logger.error(f"Background document processing failed: {e}")


@router.get("/{kb_id}/documents", response_model=List[DocumentResponse])
async def list_documents(
    kb_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents in a knowledge base."""
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    result = await db.execute(
        select(Document)
        .where(Document.knowledge_base_id == kb.id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    
    return [
        DocumentResponse(
            id=d.id,
            title=d.title,
            file_name=d.file_name,
            file_type=d.file_type,
            file_size=d.file_size,
            status=d.status.value,
            chunk_count=d.chunk_count,
            error_message=d.error_message,
            created_at=d.created_at,
            processed_at=d.processed_at,
        )
        for d in documents
    ]


@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    kb_id: int,
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get document details."""
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    result = await db.execute(
        select(Document)
        .where(Document.id == doc_id)
        .where(Document.knowledge_base_id == kb.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.id,
        title=document.title,
        file_name=document.file_name,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status.value,
        chunk_count=document.chunk_count,
        error_message=document.error_message,
        created_at=document.created_at,
        processed_at=document.processed_at,
    )


@router.delete("/{kb_id}/documents/{doc_id}", response_model=MessageResponse)
async def delete_document(
    kb_id: int,
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and its chunks."""
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    result = await db.execute(
        select(Document)
        .where(Document.id == doc_id)
        .where(Document.knowledge_base_id == kb.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete chunks
    await db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
    
    # Delete file if exists
    if document.file_path:
        try:
            Path(document.file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete file {document.file_path}: {e}")
    
    # Delete document
    await db.delete(document)
    await db.commit()
    
    logger.info(f"Document deleted: {doc_id}")
    
    return MessageResponse(message="Document deleted successfully")


# ============================================================================
# Search
# ============================================================================

@router.post("/{kb_id}/search", response_model=SearchResponse)
async def search_knowledge_base(
    kb_id: int,
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search the knowledge base using semantic similarity.
    
    Uses pgvector for fast cosine similarity search.
    """
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    search_service = get_search_service()
    
    try:
        results = await search_service.search(
            db=db,
            knowledge_base_id=kb.id,
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )
        
        return SearchResponse(
            query=request.query,
            results=[
                SearchResultResponse(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_title=r.document_title,
                    content=r.content,
                    score=r.score,
                )
                for r in results
            ],
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.get("/{kb_id}/stats", response_model=StatsResponse)
async def get_knowledge_base_stats(
    kb_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get statistics about a knowledge base."""
    customer = await get_customer_for_user(db, current_user)
    kb = await get_kb_for_customer(db, kb_id, customer.id)
    
    search_service = get_search_service()
    
    stats = await search_service.get_knowledge_base_stats(db, kb.id)
    
    return StatsResponse(**stats)
