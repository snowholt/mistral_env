"""
Unit Tests for WebRTC Signaling Endpoints.

Tests cover SDP offer/answer exchange, ICE candidate handling,
connection status, and cleanup operations.

Created for WebRTC MVP Migration - Phase B
Author: BeautyAI Framework
Date: October 15, 2025
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Test if aiortc is available
try:
    import aiortc
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

# Skip all tests if aiortc not available
pytestmark = pytest.mark.skipif(
    not AIORTC_AVAILABLE,
    reason="aiortc not installed - WebRTC tests skipped"
)


# Sample SDP offers for testing
VALID_SDP_OFFER = """v=0
o=- 123456789 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic: WMS stream
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:test
a=ice-pwd:testpassword123456789012345
a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
a=setup:actpass
a=mid:0
a=sendrecv
a=rtcp-mux
a=rtpmap:111 opus/48000/2
"""

VALID_ICE_CANDIDATE = "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host"


class TestWebRTCSignalingEndpoints:
    """Test suite for WebRTC signaling endpoints."""
    
    @pytest.fixture
    async def mock_webrtc_config(self):
        """Mock WebRTC configuration."""
        return {
            'enabled': True,
            'max_utterance_sec': 10,
            'signaling_path': '/api/v1/webrtc/voice',
            'stun_servers': ['stun:stun.l.google.com:19302'],
            'turn_servers': [],
            'ice_transport_policy': 'all',
            'debug_logging': False
        }
    
    @pytest.fixture
    async def mock_connection_pool(self):
        """Mock WebRTC connection pool."""
        pool = AsyncMock()
        pool.create_peer_connection = AsyncMock(
            return_value=(
                "v=0\r\no=- 789012 2 IN IP4 192.168.1.1\r\ns=-\r\nt=0 0\r\n",
                [{"urls": "stun:stun.l.google.com:19302"}]
            )
        )
        pool.add_ice_candidate = AsyncMock(return_value=0)
        pool.peer_exists = AsyncMock(return_value=True)
        pool.get_connection_status = AsyncMock(
            return_value={
                'connection_state': 'connected',
                'ice_connection_state': 'connected',
                'ice_gathering_state': 'complete',
                'created_at': time.time(),
                'last_activity': time.time()
            }
        )
        pool.remove_peer_connection = AsyncMock()
        pool.get_pool_stats = AsyncMock(
            return_value={
                'active_connections': 1,
                'total_connections': 1
            }
        )
        return pool
    
    @pytest.fixture
    async def mock_session_manager(self):
        """Mock WebRTC session manager."""
        manager = AsyncMock()
        manager.create_session = AsyncMock(return_value="session_xyz789")
        manager.get_session_by_peer = AsyncMock(
            return_value={
                'session_id': 'session_xyz789',
                'peer_id': 'peer_abc123',
                'language': 'ar',
                'turn_count': 0
            }
        )
        manager.cleanup_session = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    async def test_sdp_offer_valid(self, mock_webrtc_config, mock_connection_pool, mock_session_manager):
        """Test handling valid SDP offer."""
        from beautyai_inference.api.endpoints.webrtc_voice import handle_sdp_offer, SDPOfferRequest
        
        # Create request
        request = SDPOfferRequest(
            sdp=VALID_SDP_OFFER,
            type="offer",
            language="ar",
            user_id="test_user_123"
        )
        
        # Mock config manager
        with patch('beautyai_inference.api.endpoints.webrtc_voice.get_config_manager') as mock_config:
            mock_config.return_value.get_value.return_value = mock_webrtc_config
            
            # Call endpoint
            response = await handle_sdp_offer(
                request=request,
                connection_pool=mock_connection_pool,
                session_manager=mock_session_manager
            )
        
        # Assertions
        assert response.type == "answer"
        assert response.peer_id.startswith("peer_")
        assert response.session_id == "session_xyz789"
        assert len(response.ice_servers) > 0
        assert response.created_at > 0
        
        # Verify mocks called
        mock_connection_pool.create_peer_connection.assert_called_once()
        mock_session_manager.create_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sdp_offer_webrtc_disabled(self, mock_connection_pool, mock_session_manager):
        """Test SDP offer when WebRTC is disabled."""
        from beautyai_inference.api.endpoints.webrtc_voice import handle_sdp_offer, SDPOfferRequest
        from fastapi import HTTPException
        
        request = SDPOfferRequest(
            sdp=VALID_SDP_OFFER,
            type="offer"
        )
        
        # Mock config with WebRTC disabled
        with patch('beautyai_inference.api.endpoints.webrtc_voice.get_config_manager') as mock_config:
            mock_config.return_value.get_value.return_value = {'enabled': False}
            
            # Should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await handle_sdp_offer(
                    request=request,
                    connection_pool=mock_connection_pool,
                    session_manager=mock_session_manager
                )
            
            assert exc_info.value.status_code == 503
            assert "disabled" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_sdp_offer_invalid_sdp(self, mock_webrtc_config):
        """Test handling invalid SDP offer."""
        from beautyai_inference.api.endpoints.webrtc_voice import SDPOfferRequest
        from pydantic import ValidationError
        
        # Invalid SDP - missing required fields
        with pytest.raises(ValidationError) as exc_info:
            SDPOfferRequest(
                sdp="invalid sdp content",
                type="offer"
            )
        
        # Check validation error
        errors = exc_info.value.errors()
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_sdp_offer_connection_failure(self, mock_webrtc_config, mock_session_manager):
        """Test SDP offer when peer connection creation fails."""
        from beautyai_inference.api.endpoints.webrtc_voice import handle_sdp_offer, SDPOfferRequest
        from fastapi import HTTPException
        
        request = SDPOfferRequest(
            sdp=VALID_SDP_OFFER,
            type="offer"
        )
        
        # Mock connection pool with failure
        mock_pool = AsyncMock()
        mock_pool.create_peer_connection = AsyncMock(
            side_effect=Exception("Connection creation failed")
        )
        
        with patch('beautyai_inference.api.endpoints.webrtc_voice.get_config_manager') as mock_config:
            mock_config.return_value.get_value.return_value = mock_webrtc_config
            
            with pytest.raises(HTTPException) as exc_info:
                await handle_sdp_offer(
                    request=request,
                    connection_pool=mock_pool,
                    session_manager=mock_session_manager
                )
            
            assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_ice_candidate_valid(self, mock_connection_pool):
        """Test handling valid ICE candidate."""
        from beautyai_inference.api.endpoints.webrtc_voice import handle_ice_candidate, ICECandidateRequest
        
        request = ICECandidateRequest(
            peer_id="peer_abc123",
            candidate=VALID_ICE_CANDIDATE,
            sdp_mid="0",
            sdp_m_line_index=0
        )
        
        response = await handle_ice_candidate(
            request=request,
            connection_pool=mock_connection_pool
        )
        
        assert response.peer_id == "peer_abc123"
        assert response.accepted is True
        assert response.candidate_index == 0
        
        mock_connection_pool.add_ice_candidate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ice_candidate_peer_not_found(self):
        """Test ICE candidate for non-existent peer."""
        from beautyai_inference.api.endpoints.webrtc_voice import handle_ice_candidate, ICECandidateRequest
        from fastapi import HTTPException
        
        request = ICECandidateRequest(
            peer_id="peer_nonexistent",
            candidate=VALID_ICE_CANDIDATE
        )
        
        # Mock pool with peer not found
        mock_pool = AsyncMock()
        mock_pool.peer_exists = AsyncMock(return_value=False)
        
        with pytest.raises(HTTPException) as exc_info:
            await handle_ice_candidate(
                request=request,
                connection_pool=mock_pool
            )
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_ice_candidate_invalid_format(self):
        """Test ICE candidate with invalid format."""
        from beautyai_inference.api.endpoints.webrtc_voice import ICECandidateRequest
        from pydantic import ValidationError
        
        # Invalid candidate format
        with pytest.raises(ValidationError):
            ICECandidateRequest(
                peer_id="peer_abc123",
                candidate="invalid_candidate_format"
            )
    
    @pytest.mark.asyncio
    async def test_get_connection_status(self, mock_connection_pool, mock_session_manager):
        """Test getting connection status."""
        from beautyai_inference.api.endpoints.webrtc_voice import get_connection_status
        
        response = await get_connection_status(
            peer_id="peer_abc123",
            connection_pool=mock_connection_pool,
            session_manager=mock_session_manager
        )
        
        assert response.peer_id == "peer_abc123"
        assert response.session_id == "session_xyz789"
        assert response.connection_state == "connected"
        assert response.ice_connection_state == "connected"
        
        mock_connection_pool.get_connection_status.assert_called_once()
        mock_session_manager.get_session_by_peer.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_connection_status_not_found(self):
        """Test getting status for non-existent connection."""
        from beautyai_inference.api.endpoints.webrtc_voice import get_connection_status
        from fastapi import HTTPException
        
        mock_pool = AsyncMock()
        mock_pool.peer_exists = AsyncMock(return_value=False)
        
        mock_manager = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await get_connection_status(
                peer_id="peer_nonexistent",
                connection_pool=mock_pool,
                session_manager=mock_manager
            )
        
        assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_cleanup_connection(self, mock_connection_pool, mock_session_manager):
        """Test connection cleanup."""
        from beautyai_inference.api.endpoints.webrtc_voice import cleanup_connection
        
        response = await cleanup_connection(
            peer_id="peer_abc123",
            connection_pool=mock_connection_pool,
            session_manager=mock_session_manager
        )
        
        assert response.peer_id == "peer_abc123"
        assert response.cleaned_up is True
        assert "success" in response.message.lower()
        
        mock_connection_pool.remove_peer_connection.assert_called_once()
        mock_session_manager.cleanup_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_connection_not_found(self, mock_session_manager):
        """Test cleanup for non-existent connection."""
        from beautyai_inference.api.endpoints.webrtc_voice import cleanup_connection
        
        mock_pool = AsyncMock()
        mock_pool.peer_exists = AsyncMock(return_value=False)
        
        response = await cleanup_connection(
            peer_id="peer_nonexistent",
            connection_pool=mock_pool,
            session_manager=mock_session_manager
        )
        
        # Should still return success (idempotent cleanup)
        assert response.cleaned_up is True
        assert "already" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_connection_pool):
        """Test WebRTC health check endpoint."""
        from beautyai_inference.api.endpoints.webrtc_voice import webrtc_health_check
        
        with patch('beautyai_inference.api.endpoints.webrtc_voice.get_config_manager') as mock_config:
            mock_config.return_value.get_value.return_value = {
                'enabled': True
            }
            
            response = await webrtc_health_check(
                connection_pool=mock_connection_pool
            )
        
        assert response['status'] == 'healthy'
        assert response['enabled'] is True
        assert response['active_connections'] == 1
        assert 'timestamp' in response


class TestWebRTCRequestValidation:
    """Test request validation for WebRTC endpoints."""
    
    def test_sdp_offer_request_validation(self):
        """Test SDP offer request validation."""
        from beautyai_inference.api.endpoints.webrtc_voice import SDPOfferRequest
        
        # Valid request
        request = SDPOfferRequest(
            sdp=VALID_SDP_OFFER,
            type="offer",
            language="ar"
        )
        assert request.language == "ar"
        assert request.type == "offer"
        
        # Test defaults
        assert request.session_metadata == {}
    
    def test_sdp_offer_empty_sdp(self):
        """Test SDP offer with empty SDP."""
        from beautyai_inference.api.endpoints.webrtc_voice import SDPOfferRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            SDPOfferRequest(
                sdp="",
                type="offer"
            )
    
    def test_sdp_offer_invalid_language(self):
        """Test SDP offer with invalid language."""
        from beautyai_inference.api.endpoints.webrtc_voice import SDPOfferRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            SDPOfferRequest(
                sdp=VALID_SDP_OFFER,
                type="offer",
                language="invalid_lang"
            )
    
    def test_ice_candidate_request_validation(self):
        """Test ICE candidate request validation."""
        from beautyai_inference.api.endpoints.webrtc_voice import ICECandidateRequest
        
        # Valid request
        request = ICECandidateRequest(
            peer_id="peer_abc123",
            candidate=VALID_ICE_CANDIDATE,
            sdp_mid="0",
            sdp_m_line_index=0
        )
        assert request.peer_id == "peer_abc123"
        assert request.sdp_m_line_index == 0
    
    def test_ice_candidate_missing_peer_id(self):
        """Test ICE candidate without peer_id."""
        from beautyai_inference.api.endpoints.webrtc_voice import ICECandidateRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ICECandidateRequest(
                candidate=VALID_ICE_CANDIDATE
            )


class TestWebRTCConnectionPool:
    """Test WebRTC connection pool functionality."""
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self):
        """Test connection pool initialization."""
        from beautyai_inference.core.webrtc_connection_pool import WebRTCConnectionPool
        
        pool = WebRTCConnectionPool(
            max_connections=50,
            connection_timeout_seconds=300
        )
        
        assert pool.max_connections == 50
        assert pool.connection_timeout_seconds == 300
        assert not pool._running
    
    @pytest.mark.asyncio
    async def test_pool_start_stop(self):
        """Test pool start and stop."""
        from beautyai_inference.core.webrtc_connection_pool import WebRTCConnectionPool
        
        pool = WebRTCConnectionPool()
        
        await pool.start()
        assert pool._running is True
        
        await pool.stop()
        assert pool._running is False


class TestWebRTCSessionManager:
    """Test WebRTC session manager functionality."""
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test session creation."""
        from beautyai_inference.core.webrtc_session_manager import WebRTCSessionManager
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        voice_manager = VoiceSessionManager(persist_sessions=False)
        manager = WebRTCSessionManager(voice_session_manager=voice_manager)
        
        with patch('beautyai_inference.core.webrtc_session_manager.get_config_manager') as mock_config:
            mock_config.return_value.get_value.return_value = {
                'tts_voice': 'ar-SA-ZariyahNeural'
            }
            
            session_id = await manager.create_session(
                peer_id="peer_test123",
                language="ar",
                user_id="user_456"
            )
        
        assert session_id.startswith("webrtc_session_")
        assert manager.get_active_session_count() == 1
    
    @pytest.mark.asyncio
    async def test_get_session_by_peer(self):
        """Test getting session by peer_id."""
        from beautyai_inference.core.webrtc_session_manager import WebRTCSessionManager
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        voice_manager = VoiceSessionManager(persist_sessions=False)
        manager = WebRTCSessionManager(voice_session_manager=voice_manager)
        
        with patch('beautyai_inference.core.webrtc_session_manager.get_config_manager') as mock_config:
            mock_config.return_value.get_value.return_value = {
                'tts_voice': 'ar-SA-ZariyahNeural'
            }
            
            session_id = await manager.create_session(
                peer_id="peer_test123",
                language="en"
            )
            
            session_info = await manager.get_session_by_peer("peer_test123")
        
        assert session_info is not None
        assert session_info['peer_id'] == "peer_test123"
        assert session_info['language'] == "en"
    
    @pytest.mark.asyncio
    async def test_cleanup_session(self):
        """Test session cleanup."""
        from beautyai_inference.core.webrtc_session_manager import WebRTCSessionManager
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        voice_manager = VoiceSessionManager(persist_sessions=False)
        manager = WebRTCSessionManager(voice_session_manager=voice_manager)
        
        with patch('beautyai_inference.core.webrtc_session_manager.get_config_manager') as mock_config:
            mock_config.return_value.get_value.return_value = {
                'tts_voice': 'ar-SA-ZariyahNeural'
            }
            
            await manager.create_session(peer_id="peer_test123", language="ar")
            assert manager.get_active_session_count() == 1
            
            await manager.cleanup_session("peer_test123")
            assert manager.get_active_session_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
