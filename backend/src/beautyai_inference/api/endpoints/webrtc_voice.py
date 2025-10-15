"""
WebRTC Voice Signaling Endpoints.

This module provides FastAPI endpoints for WebRTC voice-to-voice signaling,
including SDP offer/answer exchange, ICE candidate handling, and connection lifecycle.

Created for WebRTC MVP Migration - Phase B
Author: BeautyAI Framework
Date: October 15, 2025
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, WebSocket, status
from pydantic import BaseModel, Field, validator

from ...core.webrtc_connection_pool import WebRTCConnectionPool, get_webrtc_pool
from ...core.webrtc_session_manager import WebRTCSessionManager, get_webrtc_session_manager
from ...core.config_manager import get_config_manager

logger = logging.getLogger(__name__)

# Create router with tags for documentation
webrtc_voice_router = APIRouter(
    prefix="/api/v1/webrtc/voice",
    tags=["webrtc-voice"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class SDPOfferRequest(BaseModel):
    """Request model for SDP offer exchange."""
    
    sdp: str = Field(
        ..., 
        description="Session Description Protocol offer in string format",
        min_length=10,
        max_length=50000
    )
    type: str = Field(
        default="offer",
        description="SDP type (must be 'offer')",
        pattern="^offer$"
    )
    language: Optional[str] = Field(
        default="ar",
        description="Conversation language (ar, en)",
        pattern="^(ar|en)$"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier",
        max_length=255
    )
    session_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional session metadata"
    )
    
    @validator('sdp')
    def validate_sdp_content(cls, v):
        """Validate SDP contains required fields."""
        if not v or len(v.strip()) == 0:
            raise ValueError("SDP offer cannot be empty")
        
        # Basic SDP validation - check for required lines
        required_fields = ['v=', 'm=', 'c=']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"SDP missing required field: {field}")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "sdp": "v=0\r\no=- 123456 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n...",
                "type": "offer",
                "language": "ar",
                "user_id": "user_123",
                "session_metadata": {"client": "web", "version": "1.0"}
            }
        }


class SDPAnswerResponse(BaseModel):
    """Response model for SDP answer."""
    
    sdp: str = Field(
        ...,
        description="Session Description Protocol answer"
    )
    type: str = Field(
        default="answer",
        description="SDP type"
    )
    peer_id: str = Field(
        ...,
        description="Unique peer connection identifier"
    )
    session_id: str = Field(
        ...,
        description="Voice session identifier"
    )
    ice_servers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="ICE servers configuration (STUN/TURN)"
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Timestamp when answer was created"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sdp": "v=0\r\no=- 789012 2 IN IP4 192.168.1.1\r\ns=-\r\nt=0 0\r\n...",
                "type": "answer",
                "peer_id": "peer_abc123",
                "session_id": "session_xyz789",
                "ice_servers": [
                    {"urls": "stun:stun.l.google.com:19302"}
                ],
                "created_at": 1697385600.0
            }
        }


class ICECandidateRequest(BaseModel):
    """Request model for ICE candidate exchange."""
    
    peer_id: str = Field(
        ...,
        description="Peer connection identifier",
        min_length=1,
        max_length=255
    )
    candidate: str = Field(
        ...,
        description="ICE candidate string",
        min_length=1,
        max_length=2000
    )
    sdp_mid: Optional[str] = Field(
        default=None,
        description="Media stream identification tag"
    )
    sdp_m_line_index: Optional[int] = Field(
        default=None,
        description="Media line index",
        ge=0
    )
    
    @validator('candidate')
    def validate_candidate_format(cls, v):
        """Validate ICE candidate format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("ICE candidate cannot be empty")
        
        # Basic validation - check for 'candidate:' prefix
        if not v.strip().startswith('candidate:'):
            raise ValueError("Invalid ICE candidate format (must start with 'candidate:')")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "peer_id": "peer_abc123",
                "candidate": "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host",
                "sdp_mid": "0",
                "sdp_m_line_index": 0
            }
        }


class ICECandidateResponse(BaseModel):
    """Response model for ICE candidate acknowledgment."""
    
    peer_id: str = Field(..., description="Peer connection identifier")
    candidate_index: int = Field(..., description="Candidate index")
    accepted: bool = Field(default=True, description="Whether candidate was accepted")
    message: str = Field(default="ICE candidate accepted", description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "peer_id": "peer_abc123",
                "candidate_index": 0,
                "accepted": True,
                "message": "ICE candidate accepted"
            }
        }


class ConnectionStatusResponse(BaseModel):
    """Response model for connection status."""
    
    peer_id: str
    session_id: str
    connection_state: str
    ice_connection_state: str
    ice_gathering_state: str
    created_at: float
    last_activity: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "peer_id": "peer_abc123",
                "session_id": "session_xyz789",
                "connection_state": "connected",
                "ice_connection_state": "connected",
                "ice_gathering_state": "complete",
                "created_at": 1697385600.0,
                "last_activity": 1697385620.0
            }
        }


class CleanupResponse(BaseModel):
    """Response model for connection cleanup."""
    
    peer_id: str
    cleaned_up: bool
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "peer_id": "peer_abc123",
                "cleaned_up": True,
                "message": "Connection cleaned up successfully"
            }
        }


# ============================================================================
# Endpoints
# ============================================================================

@webrtc_voice_router.post(
    "/offer",
    response_model=SDPAnswerResponse,
    summary="Handle SDP Offer",
    description="Process SDP offer from client and return SDP answer with peer_id",
    status_code=status.HTTP_200_OK
)
async def handle_sdp_offer(
    request: SDPOfferRequest,
    connection_pool: WebRTCConnectionPool = Depends(get_webrtc_pool),
    session_manager: WebRTCSessionManager = Depends(get_webrtc_session_manager)
) -> SDPAnswerResponse:
    """
    Handle SDP offer from WebRTC client.
    
    This endpoint:
    1. Validates the SDP offer
    2. Creates a new RTCPeerConnection
    3. Generates SDP answer
    4. Creates voice session
    5. Returns answer with peer_id for future communication
    
    Args:
        request: SDP offer request with optional metadata
        connection_pool: WebRTC connection pool dependency
        session_manager: WebRTC session manager dependency
        
    Returns:
        SDPAnswerResponse with SDP answer and identifiers
        
    Raises:
        HTTPException: If offer processing fails
    """
    try:
        # Generate unique peer_id
        peer_id = f"peer_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"[WebRTC] Handling SDP offer for peer_id={peer_id}, language={request.language}")
        
        # Get WebRTC configuration from environment
        import os
        webrtc_enabled = os.getenv('WEBRTC_ENABLED', '1') == '1'
        
        # Check if WebRTC is enabled
        if not webrtc_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="WebRTC voice service is currently disabled. Please enable it in configuration."
            )
        
        # Create RTCPeerConnection and process offer
        try:
            answer_sdp, ice_servers = await connection_pool.create_peer_connection(
                peer_id=peer_id,
                offer_sdp=request.sdp,
                user_id=request.user_id
            )
        except Exception as e:
            logger.error(f"[WebRTC] Failed to create peer connection for {peer_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process SDP offer: {str(e)}"
            )
        
        # Create voice session
        try:
            session_id = await session_manager.create_session(
                peer_id=peer_id,
                language=request.language or "ar",
                user_id=request.user_id,
                metadata=request.session_metadata
            )
        except Exception as e:
            logger.error(f"[WebRTC] Failed to create voice session for {peer_id}: {e}", exc_info=True)
            # Clean up peer connection on session failure
            await connection_pool.remove_peer_connection(peer_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create voice session: {str(e)}"
            )
        
        logger.info(f"[WebRTC] Successfully created peer connection and session: peer_id={peer_id}, session_id={session_id}")
        
        # Return SDP answer with identifiers
        return SDPAnswerResponse(
            sdp=answer_sdp,
            type="answer",
            peer_id=peer_id,
            session_id=session_id,
            ice_servers=ice_servers,
            created_at=time.time()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WebRTC] Unexpected error handling SDP offer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@webrtc_voice_router.post(
    "/ice",
    response_model=ICECandidateResponse,
    summary="Handle ICE Candidate",
    description="Process ICE candidate from client for trickle ICE",
    status_code=status.HTTP_200_OK
)
async def handle_ice_candidate(
    request: ICECandidateRequest,
    connection_pool: WebRTCConnectionPool = Depends(get_webrtc_pool)
) -> ICECandidateResponse:
    """
    Handle ICE candidate from WebRTC client.
    
    This endpoint supports trickle ICE by accepting candidates
    after the initial offer/answer exchange.
    
    Args:
        request: ICE candidate request
        connection_pool: WebRTC connection pool dependency
        
    Returns:
        ICECandidateResponse with acknowledgment
        
    Raises:
        HTTPException: If candidate processing fails
    """
    try:
        logger.debug(f"[WebRTC] Handling ICE candidate for peer_id={request.peer_id}")
        
        # Check if peer connection exists
        if not await connection_pool.peer_exists(request.peer_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Peer connection not found: {request.peer_id}"
            )
        
        # Add ICE candidate to peer connection
        try:
            candidate_index = await connection_pool.add_ice_candidate(
                peer_id=request.peer_id,
                candidate=request.candidate,
                sdp_mid=request.sdp_mid,
                sdp_m_line_index=request.sdp_m_line_index
            )
        except Exception as e:
            logger.error(f"[WebRTC] Failed to add ICE candidate for {request.peer_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ICE candidate: {str(e)}"
            )
        
        logger.debug(f"[WebRTC] ICE candidate added successfully: peer_id={request.peer_id}, index={candidate_index}")
        
        return ICECandidateResponse(
            peer_id=request.peer_id,
            candidate_index=candidate_index,
            accepted=True,
            message="ICE candidate accepted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WebRTC] Unexpected error handling ICE candidate: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@webrtc_voice_router.get(
    "/{peer_id}/status",
    response_model=ConnectionStatusResponse,
    summary="Get Connection Status",
    description="Retrieve current status of a WebRTC peer connection",
    status_code=status.HTTP_200_OK
)
async def get_connection_status(
    peer_id: str,
    connection_pool: WebRTCConnectionPool = Depends(get_webrtc_pool),
    session_manager: WebRTCSessionManager = Depends(get_webrtc_session_manager)
) -> ConnectionStatusResponse:
    """
    Get current status of a WebRTC peer connection.
    
    Args:
        peer_id: Peer connection identifier
        connection_pool: WebRTC connection pool dependency
        session_manager: WebRTC session manager dependency
        
    Returns:
        ConnectionStatusResponse with current status
        
    Raises:
        HTTPException: If peer connection not found
    """
    try:
        # Check if peer connection exists
        if not await connection_pool.peer_exists(peer_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Peer connection not found: {peer_id}"
            )
        
        # Get connection status
        status_info = await connection_pool.get_connection_status(peer_id)
        
        # Get session info
        session_info = await session_manager.get_session_by_peer(peer_id)
        
        return ConnectionStatusResponse(
            peer_id=peer_id,
            session_id=session_info.get('session_id', 'unknown'),
            connection_state=status_info.get('connection_state', 'unknown'),
            ice_connection_state=status_info.get('ice_connection_state', 'unknown'),
            ice_gathering_state=status_info.get('ice_gathering_state', 'unknown'),
            created_at=status_info.get('created_at', time.time()),
            last_activity=status_info.get('last_activity', time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WebRTC] Error getting connection status for {peer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@webrtc_voice_router.delete(
    "/{peer_id}",
    response_model=CleanupResponse,
    summary="Cleanup Connection",
    description="Close and cleanup a WebRTC peer connection and associated session",
    status_code=status.HTTP_200_OK
)
async def cleanup_connection(
    peer_id: str,
    connection_pool: WebRTCConnectionPool = Depends(get_webrtc_pool),
    session_manager: WebRTCSessionManager = Depends(get_webrtc_session_manager)
) -> CleanupResponse:
    """
    Cleanup a WebRTC peer connection and associated voice session.
    
    This endpoint:
    1. Closes the RTCPeerConnection
    2. Removes peer from connection pool
    3. Cleans up voice session
    4. Releases resources
    
    Args:
        peer_id: Peer connection identifier
        connection_pool: WebRTC connection pool dependency
        session_manager: WebRTC session manager dependency
        
    Returns:
        CleanupResponse with cleanup confirmation
        
    Raises:
        HTTPException: If cleanup fails
    """
    try:
        logger.info(f"[WebRTC] Cleaning up connection: peer_id={peer_id}")
        
        # Check if peer connection exists
        if not await connection_pool.peer_exists(peer_id):
            # Connection already cleaned up or never existed
            logger.warning(f"[WebRTC] Peer connection not found during cleanup: {peer_id}")
            return CleanupResponse(
                peer_id=peer_id,
                cleaned_up=True,
                message="Connection already cleaned up or not found"
            )
        
        # Remove peer connection
        try:
            await connection_pool.remove_peer_connection(peer_id)
        except Exception as e:
            logger.error(f"[WebRTC] Error removing peer connection {peer_id}: {e}", exc_info=True)
            # Continue with session cleanup even if peer removal fails
        
        # Cleanup voice session
        try:
            await session_manager.cleanup_session(peer_id)
        except Exception as e:
            logger.error(f"[WebRTC] Error cleaning up session for {peer_id}: {e}", exc_info=True)
            # Don't raise - peer connection is already cleaned up
        
        logger.info(f"[WebRTC] Successfully cleaned up connection: peer_id={peer_id}")
        
        return CleanupResponse(
            peer_id=peer_id,
            cleaned_up=True,
            message="Connection cleaned up successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WebRTC] Unexpected error during cleanup for {peer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during cleanup: {str(e)}"
        )


@webrtc_voice_router.get(
    "/health",
    summary="WebRTC Health Check",
    description="Check if WebRTC signaling service is healthy",
    status_code=status.HTTP_200_OK
)
async def webrtc_health_check(
    connection_pool: WebRTCConnectionPool = Depends(get_webrtc_pool)
) -> Dict[str, Any]:
    """
    Health check endpoint for WebRTC signaling service.
    
    Returns:
        Health status including active connections count
    """
    try:
        # Get WebRTC configuration from environment
        import os
        webrtc_enabled = os.getenv('WEBRTC_ENABLED', '1') == '1'
        
        pool_stats = await connection_pool.get_pool_stats()
        
        return {
            "status": "healthy",
            "enabled": webrtc_enabled,
            "active_connections": pool_stats.get('active_connections', 0),
            "total_connections": pool_stats.get('total_connections', 0),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"[WebRTC] Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


# Export router
__all__ = ['webrtc_voice_router']
