"""
Unit tests for Simple Voice WebSocket Endpoint.

Tests the SimpleVoiceWebSocketManager and websocket_simple_voice_chat endpoint
for ultra-fast voice conversation functionality.

Author: BeautyAI Framework
Date: 2025-07-23
"""

import asyncio
import json
import pytest
import base64
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

# Import the components to test
from beautyai_inference.api.endpoints.websocket_simple_voice import (
    SimpleVoiceWebSocketManager,
    simple_ws_manager,
    simple_voice_connections,
    websocket_simple_voice_router
)
from beautyai_inference.api.app import app


class TestSimpleVoiceWebSocketManager:
    """Test cases for SimpleVoiceWebSocketManager."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear connections before each test
        simple_voice_connections.clear()
        yield
        # Clear connections after each test
        simple_voice_connections.clear()
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager instance for testing."""
        return SimpleVoiceWebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket object."""
        websocket = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        return websocket
    
    @pytest.mark.asyncio
    async def test_connect_success(self, manager, mock_websocket):
        """Test successful WebSocket connection establishment."""
        with patch.object(manager, '_ensure_service_initialized', new_callable=AsyncMock):
            result = await manager.connect(
                websocket=mock_websocket,
                connection_id="test-123",
                language="ar",
                voice_type="female",
                session_id="test-session"
            )
            
            assert result is True
            mock_websocket.accept.assert_called_once()
            mock_websocket.send_text.assert_called_once()
            
            # Check that connection was stored
            assert "test-123" in simple_voice_connections
            connection = simple_voice_connections["test-123"]
            assert connection["language"] == "ar"
            assert connection["voice_type"] == "female"
            assert connection["session_id"] == "test-session"
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, manager, mock_websocket):
        """Test WebSocket connection failure handling."""
        # Make accept() raise an exception
        mock_websocket.accept.side_effect = Exception("Connection failed")
        
        with patch.object(manager, '_ensure_service_initialized', new_callable=AsyncMock):
            result = await manager.connect(
                websocket=mock_websocket,
                connection_id="test-fail",
                language="en",
                voice_type="male"
            )
            
            assert result is False
            assert "test-fail" not in simple_voice_connections
    
    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """Test WebSocket disconnection and cleanup."""
        # Ensure clean state
        simple_voice_connections.clear()
        
        # First establish a connection
        connection_id = "test-disconnect"
        simple_voice_connections[connection_id] = {
            "websocket": mock_websocket,
            "language": "ar",
            "voice_type": "female",
            "session_id": "test-session",
            "connected_at": 1234567890,
            "message_count": 5
        }
        
        with patch.object(manager, '_cleanup_service', new_callable=AsyncMock) as mock_cleanup:
            await manager.disconnect(connection_id)
            
            # Check that connection was removed
            assert connection_id not in simple_voice_connections
            
            # Cleanup should be called since no connections remain
            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, manager, mock_websocket):
        """Test successful message sending."""
        connection_id = "test-send"
        simple_voice_connections[connection_id] = {
            "websocket": mock_websocket
        }
        
        # Mock WebSocket state
        from starlette.websockets import WebSocketState
        mock_websocket.client_state = WebSocketState.CONNECTED
        
        message = {"type": "test", "data": "hello"}
        result = await manager.send_message(connection_id, message)
        
        assert result is True
        mock_websocket.send_text.assert_called_once()
        
        # Check that the message was properly JSON encoded
        sent_data = mock_websocket.send_text.call_args[0][0]
        assert json.loads(sent_data) == message
    
    @pytest.mark.asyncio
    async def test_send_message_nonexistent_connection(self, manager):
        """Test sending message to non-existent connection."""
        result = await manager.send_message("nonexistent", {"type": "test"})
        assert result is False
    
    @pytest.mark.asyncio
    async def test_process_audio_message_success(self, manager, mock_websocket):
        """Test successful audio message processing."""
        connection_id = "test-audio"
        simple_voice_connections[connection_id] = {
            "websocket": mock_websocket,
            "language": "ar",
            "voice_type": "female",
            "session_id": "test-session",
            "message_count": 0
        }
        
        # Mock the _mock_voice_processing method
        with patch.object(manager, '_mock_voice_processing', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "success": True,
                "transcription": "مرحبا",
                "response_text": "أهلا وسهلا",
                "audio_base64": "dGVzdA=="  # base64 encoded "test"
            }
            
            with patch.object(manager, 'send_message', new_callable=AsyncMock) as mock_send:
                audio_data = b"fake_audio_data"
                result = await manager.process_audio_message(connection_id, audio_data)
                
                assert result["success"] is True
                assert "processing_time" in result
                
                # Check that message count was incremented
                assert simple_voice_connections[connection_id]["message_count"] == 1
                
                # Check that messages were sent (processing_started and voice_response)
                assert mock_send.call_count == 2
    
    @pytest.mark.asyncio
    async def test_process_audio_message_failure(self, manager, mock_websocket):
        """Test audio message processing failure."""
        connection_id = "test-audio-fail"
        simple_voice_connections[connection_id] = {
            "websocket": mock_websocket,
            "language": "en",
            "voice_type": "male",
            "session_id": "test-session",
            "message_count": 0
        }
        
        # Mock the _mock_voice_processing method to return failure
        with patch.object(manager, '_mock_voice_processing', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "success": False,
                "error": "Processing failed"
            }
            
            with patch.object(manager, 'send_message', new_callable=AsyncMock) as mock_send:
                audio_data = b"fake_audio_data"
                result = await manager.process_audio_message(connection_id, audio_data)
                
                assert result["success"] is False
                assert "error" in result
                
                # Check that error message was sent
                error_call = [call for call in mock_send.call_args_list if "error" in str(call)]
                assert len(error_call) > 0
    
    @pytest.mark.asyncio
    async def test_mock_voice_processing_arabic(self, manager):
        """Test mock voice processing for Arabic language."""
        with patch.object(manager.voice_service, 'text_to_speech', new_callable=AsyncMock) as mock_tts:
            # Create a temporary audio file for testing
            temp_audio = Path("/tmp/test_audio.wav")
            temp_audio.write_bytes(b"fake_audio_content")
            mock_tts.return_value = temp_audio
            
            try:
                result = await manager._mock_voice_processing(
                    audio_path="/fake/path.wav",
                    language="ar",
                    voice_type="female"
                )
                
                assert result["success"] is True
                assert "مرحبا" in result["transcription"]
                assert "أهلاً وسهلاً" in result["response_text"]
                assert "audio_base64" in result
                assert "audio_size_bytes" in result
                
                # Check that TTS was called with correct parameters
                mock_tts.assert_called_once()
                call_args = mock_tts.call_args
                assert call_args[1]["language"] == "ar"
                assert call_args[1]["gender"] == "female"
                
            finally:
                # Clean up temp file
                if temp_audio.exists():
                    temp_audio.unlink()
    
    @pytest.mark.asyncio
    async def test_mock_voice_processing_english(self, manager):
        """Test mock voice processing for English language."""
        with patch.object(manager.voice_service, 'text_to_speech', new_callable=AsyncMock) as mock_tts:
            # Create a temporary audio file for testing
            temp_audio = Path("/tmp/test_audio_en.wav")
            temp_audio.write_bytes(b"fake_audio_content_english")
            mock_tts.return_value = temp_audio
            
            try:
                result = await manager._mock_voice_processing(
                    audio_path="/fake/path.wav",
                    language="en",
                    voice_type="male"
                )
                
                assert result["success"] is True
                assert "Hello" in result["transcription"]
                assert "Hello!" in result["response_text"]
                assert "audio_base64" in result
                
                # Check that TTS was called with correct parameters
                mock_tts.assert_called_once()
                call_args = mock_tts.call_args
                assert call_args[1]["language"] == "en"
                assert call_args[1]["gender"] == "male"
                
            finally:
                # Clean up temp file
                if temp_audio.exists():
                    temp_audio.unlink()


class TestSimpleVoiceWebSocketEndpoint:
    """Test cases for the WebSocket endpoint itself."""
    
    def test_status_endpoint(self):
        """Test the status endpoint returns correct information."""
        client = TestClient(app)
        response = client.get("/ws/simple-voice-chat/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "simple_voice_chat"
        assert data["status"] == "available"
        assert "active_connections" in data
        assert "performance" in data
        assert "features" in data
        assert "audio_formats" in data
        assert "comparison_with_advanced" in data
        
        # Check performance specs
        performance = data["performance"]
        assert performance["target_response_time"] == "< 2 seconds"
        assert performance["supported_languages"] == ["ar", "en"]
        assert performance["voice_types"] == ["male", "female"]
        assert performance["engine"] == "Edge TTS via SimpleVoiceService"
        
        # Check comparison data
        comparison = data["comparison_with_advanced"]
        assert "simple_endpoint" in comparison
        assert "advanced_endpoint" in comparison
        assert comparison["simple_endpoint"]["route"] == "/ws/simple-voice-chat"
        assert comparison["advanced_endpoint"]["route"] == "/ws/voice-conversation"
    
    @pytest.mark.asyncio
    async def test_websocket_connection_validation(self):
        """Test WebSocket connection with parameter validation."""
        # This is a more complex test that would require WebSocket test client
        # For now, we'll test the validation logic through the manager
        pass


class TestIntegrationWithSimpleVoiceService:
    """Integration tests with SimpleVoiceService."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test that the manager properly initializes the voice service."""
        manager = SimpleVoiceWebSocketManager()
        
        with patch.object(manager.voice_service, 'initialize', new_callable=AsyncMock) as mock_init:
            await manager._ensure_service_initialized()
            mock_init.assert_called_once()
            assert manager._service_initialized is True
            
            # Second call should not reinitialize
            await manager._ensure_service_initialized()
            assert mock_init.call_count == 1  # Still just one call
    
    @pytest.mark.asyncio
    async def test_service_cleanup(self):
        """Test that the manager properly cleans up the voice service."""
        manager = SimpleVoiceWebSocketManager()
        manager._service_initialized = True
        
        # Ensure no connections exist to trigger cleanup
        simple_voice_connections.clear()
        
        with patch.object(manager.voice_service, 'cleanup', new_callable=AsyncMock) as mock_cleanup:
            await manager._cleanup_service()
            
            # Should wait and then cleanup
            mock_cleanup.assert_called_once()
            assert manager._service_initialized is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
