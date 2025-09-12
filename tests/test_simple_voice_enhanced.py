"""
Enhanced Simple Voice WebSocket Tests

This test suite validates the new features implemented in the Simple Voice
WebSocket endpoint, including persistent model management, adaptive VAD,
and session management.

Author: BeautyAI Framework
Date: 2025-01-23
"""

import asyncio
import json
import logging
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPersistentModelManager:
    """Test the PersistentModelManager functionality."""
    
    @pytest.mark.asyncio
    async def test_persistent_model_manager_initialization(self):
        """Test that PersistentModelManager initializes correctly."""
        from beautyai_inference.core.persistent_model_manager import PersistentModelManager
        
        manager = PersistentModelManager()
        
        # Test that the manager has the expected attributes
        assert hasattr(manager, '_initialized')
        assert hasattr(manager, '_preloaded_models')
        assert hasattr(manager, '_model_manager')
        
        # Test initialization (simplified test without mocking)
        assert manager._initialized is False  # Not initialized until preload_models() is called
        
        # Test that we can call the main methods
        assert hasattr(manager, 'preload_models')
        assert hasattr(manager, 'get_whisper_model')
        assert hasattr(manager, 'get_llm_model')
        assert hasattr(manager, 'is_initialized')
        assert hasattr(manager, 'check_models_ready')
    
    @pytest.mark.asyncio
    async def test_persistent_model_preloading(self):
        """Test model preloading functionality."""
        from beautyai_inference.core.persistent_model_manager import PersistentModelManager
        
        manager = PersistentModelManager()
        
        with patch('beautyai_inference.core.persistent_model_manager.ConfigurationManager'):
            with patch.object(manager, '_preload_whisper_model') as mock_whisper:
                with patch.object(manager, '_preload_llm_model') as mock_llm:
                    mock_whisper.return_value = True
                    mock_llm.return_value = True
                    
                    await manager.initialize()
                    await manager.preload_models()
                    
                    # Verify preloading was attempted
                    mock_whisper.assert_called_once()
                    mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_health_monitoring(self):
        """Test model health monitoring."""
        from beautyai_inference.core.persistent_model_manager import PersistentModelManager
        
        manager = PersistentModelManager()
        manager._initialized = True
        manager._preloaded_models = {"whisper": MagicMock(), "llm": MagicMock()}
        
        health_status = await manager.get_health_status()
        
        assert "status" in health_status
        assert "models" in health_status
        assert "system_metrics" in health_status


class TestAdaptiveVAD:
    """Test the adaptive VAD functionality."""
    
    @pytest.mark.asyncio
    async def test_vad_configuration(self):
        """Test VAD configuration with adaptive features."""
        from beautyai_inference.services.voice.vad_service import VADConfig
        
        config = VADConfig(
            chunk_size_ms=30,
            silence_threshold_ms=500,
            adaptive_threshold=True,
            noise_adaptation=True,
            arabic_speech_threshold=0.45
        )
        
        assert config.adaptive_threshold is True
        assert config.noise_adaptation is True
        assert config.arabic_speech_threshold == 0.45
    
    @pytest.mark.asyncio
    async def test_vad_service_initialization(self):
        """Test VAD service initialization with adaptive config."""
        from beautyai_inference.services.voice.vad_service import VADConfig, initialize_vad_service
        
        vad_config = VADConfig(
            chunk_size_ms=30,
            adaptive_threshold=True,
            energy_threshold=0.01
        )
        
        with patch('beautyai_inference.services.voice.vad_service.RealTimeVADService') as mock_vad:
            mock_instance = AsyncMock()
            mock_vad.return_value = mock_instance
            mock_instance.initialize.return_value = True
            
            success = await initialize_vad_service(vad_config)
            assert success is True
            mock_vad.assert_called_once_with(vad_config)


class TestVoiceSessionManager:
    """Test the voice session management functionality."""
    
    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test creating a new voice session."""
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            session_manager = VoiceSessionManager(
                persist_sessions=True,
                session_dir=Path(temp_dir)
            )
            
            session = await session_manager.create_session(
                connection_id="test_conn_123",
                language="ar",
                voice_type="female",
                user_id="test_user"
            )
            
            assert session.connection_id == "test_conn_123"
            assert session.language == "ar"
            assert session.voice_type == "female"
            assert session.user_id == "test_user"
            assert session.turn_count == 0
    
    @pytest.mark.asyncio
    async def test_conversation_turn_tracking(self):
        """Test adding conversation turns to a session."""
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        session_manager = VoiceSessionManager(persist_sessions=False)
        
        session = await session_manager.create_session(
            connection_id="test_conn_123",
            language="ar",
            voice_type="female"
        )
        
        # Add a conversation turn
        success = await session_manager.add_conversation_turn(
            session_id=session.session_id,
            user_input="مرحبا",
            ai_response="أهلا وسهلا",
            processing_time_ms=1500,
            transcription_quality="ok"
        )
        
        assert success is True
        assert session.turn_count == 1
        assert len(session.conversation_history) == 1
        
        turn = session.conversation_history[0]
        assert turn.user_input == "مرحبا"
        assert turn.ai_response == "أهلا وسهلا"
        assert turn.processing_time_ms == 1500
    
    @pytest.mark.asyncio
    async def test_conversation_context_retrieval(self):
        """Test retrieving conversation context."""
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        session_manager = VoiceSessionManager(persist_sessions=False)
        
        session = await session_manager.create_session(
            connection_id="test_conn_123",
            language="en",
            voice_type="female"
        )
        
        # Add multiple turns
        await session_manager.add_conversation_turn(
            session_id=session.session_id,
            user_input="Hello",
            ai_response="Hi there!",
            processing_time_ms=1000
        )
        
        await session_manager.add_conversation_turn(
            session_id=session.session_id,
            user_input="How are you?",
            ai_response="I'm doing well, thank you!",
            processing_time_ms=1200
        )
        
        context = await session_manager.get_conversation_context(session.session_id)
        
        assert "Hello" in context
        assert "Hi there!" in context
        assert "How are you?" in context
        assert "I'm doing well, thank you!" in context
    
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test session persistence to disk."""
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            
            # Create session manager with persistence
            session_manager = VoiceSessionManager(
                persist_sessions=True,
                session_dir=session_dir
            )
            
            session = await session_manager.create_session(
                connection_id="test_conn_123",
                language="ar",
                voice_type="male"
            )
            
            await session_manager.add_conversation_turn(
                session_id=session.session_id,
                user_input="مرحبا",
                ai_response="أهلا وسهلا",
                processing_time_ms=1500
            )
            
            # Check that session file was created
            session_file = session_dir / f"{session.session_id}.json"
            assert session_file.exists()
            
            # Load and verify session data
            with open(session_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert saved_data["session_id"] == session.session_id
            assert saved_data["language"] == "ar"
            assert saved_data["turn_count"] == 1
            assert len(saved_data["conversation_history"]) == 1
    
    @pytest.mark.asyncio
    async def test_session_expiration_cleanup(self):
        """Test automatic cleanup of expired sessions."""
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        session_manager = VoiceSessionManager(persist_sessions=False)
        session_manager.session_timeout_minutes = 0.01  # 0.6 seconds for testing
        
        session = await session_manager.create_session(
            connection_id="test_conn_123",
            language="en",
            voice_type="female"
        )
        
        # Wait for session to expire
        await asyncio.sleep(1)
        
        # Run cleanup
        cleaned_count = await session_manager.cleanup_expired_sessions()
        
        assert cleaned_count == 1
        assert session.session_id not in session_manager.active_sessions


class TestSimpleVoiceServiceEnhancements:
    """Test enhancements to the Simple Voice Service."""
    
    @pytest.mark.asyncio
    async def test_persistent_model_manager_integration(self):
        """Test integration with persistent model manager."""
        from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
        
        service = SimpleVoiceService()
        
        # Mock persistent model manager
        mock_persistent_manager = MagicMock()
        mock_whisper_engine = MagicMock()
        mock_persistent_manager.get_whisper_model = AsyncMock(return_value=mock_whisper_engine)
        
        service.set_persistent_model_manager(mock_persistent_manager)
        
        assert service.persistent_model_manager == mock_persistent_manager
    
    @pytest.mark.asyncio
    async def test_conversation_context_integration(self):
        """Test conversation context integration in voice processing."""
        from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
        
        service = SimpleVoiceService()
        
        # Mock the necessary components
        with patch.object(service, '_transcribe_audio') as mock_transcribe:
            with patch.object(service, '_generate_chat_response') as mock_chat:
                with patch.object(service, '_synthesize_speech') as mock_tts:
                    mock_transcribe.return_value = "Hello"
                    mock_chat.return_value = "Hi there!"
                    mock_tts.return_value = Path("/tmp/test.wav")
                    
                    # Test with conversation context
                    result = await service.process_voice_message(
                        audio_data=b"fake_audio",
                        audio_format="wav",
                        language="en",
                        gender="female",
                        conversation_context="User: Hello\nAI: Hi!\nUser: How are you?\nAI: I'm well!"
                    )
                    
                    # Verify that context was passed to chat response generation
                    mock_chat.assert_called_once()
                    call_args = mock_chat.call_args
                    assert "conversation_context" in call_args.kwargs
                    assert "How are you?" in call_args.kwargs["conversation_context"]


class TestWebSocketEnhancements:
    """Test WebSocket endpoint enhancements."""
    
    @pytest.mark.asyncio
    async def test_websocket_manager_initialization(self):
        """Test enhanced WebSocket manager initialization."""
        from beautyai_inference.api.endpoints.websocket_simple_voice import SimpleVoiceWebSocketManager
        
        manager = SimpleVoiceWebSocketManager()
        
        # Check that new components are initialized
        assert hasattr(manager, 'persistent_model_manager')
        assert hasattr(manager, 'session_manager')
        assert hasattr(manager, '_persistent_models_initialized')
        assert hasattr(manager, '_cleanup_task')
        assert hasattr(manager, '_cleanup_interval')
    
    @pytest.mark.asyncio
    async def test_session_integration_in_connection(self):
        """Test session creation during WebSocket connection."""
        from beautyai_inference.api.endpoints.websocket_simple_voice import SimpleVoiceWebSocketManager
        
        manager = SimpleVoiceWebSocketManager()
        
        # Mock the necessary components
        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()
        
        with patch.object(manager, '_ensure_service_initialized') as mock_init:
            with patch.object(manager, '_get_connection_pool') as mock_pool:
                with patch.object(manager, 'send_message') as mock_send:
                    mock_init.return_value = None
                    
                    mock_pool_instance = MagicMock()
                    mock_pool_instance.register_websocket = AsyncMock(return_value="pool_conn_123")
                    mock_pool_instance.get_connection.return_value = MagicMock()
                    mock_pool.return_value = mock_pool_instance
                    
                    mock_send.return_value = True
                    
                    # Test connection with session creation
                    success = await manager.connect(
                        websocket=mock_websocket,
                        connection_id="test_conn_123",
                        language="ar",
                        voice_type="female",
                        session_id="test_session_456"
                    )
                    
                    assert success is True
                    mock_websocket.accept.assert_called_once()


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for the complete enhanced system."""
    
    @pytest.mark.asyncio
    async def test_complete_voice_conversation_flow(self):
        """Test a complete voice conversation flow with all enhancements."""
        from beautyai_inference.api.endpoints.websocket_simple_voice import SimpleVoiceWebSocketManager
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        # Create managers
        ws_manager = SimpleVoiceWebSocketManager()
        session_manager = VoiceSessionManager(persist_sessions=False)
        
        # Create a voice session
        session = await session_manager.create_session(
            connection_id="test_conn_123",
            language="en",
            voice_type="female"
        )
        
        # Simulate multiple conversation turns
        turns = [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well, thank you!"),
            ("What's the weather like?", "I don't have weather information, but I'm here to chat!")
        ]
        
        for i, (user_input, ai_response) in enumerate(turns):
            success = await session_manager.add_conversation_turn(
                session_id=session.session_id,
                user_input=user_input,
                ai_response=ai_response,
                processing_time_ms=1000 + i * 200,
                transcription_quality="ok"
            )
            assert success is True
        
        # Verify session state
        assert session.turn_count == 3
        assert len(session.conversation_history) == 3
        assert session.average_response_time_ms > 0
        assert session.transcription_success_rate == 1.0
        
        # Get conversation context
        context = await session_manager.get_conversation_context(session.session_id)
        assert "Hello" in context
        assert "weather" in context
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test that performance metrics are collected correctly."""
        from beautyai_inference.core.voice_session_manager import VoiceSessionManager
        
        session_manager = VoiceSessionManager(persist_sessions=False)
        
        # Create multiple sessions with different characteristics
        session1 = await session_manager.create_session("conn1", "ar", "female")
        session2 = await session_manager.create_session("conn2", "en", "male")
        
        # Add turns with different processing times and quality
        await session_manager.add_conversation_turn(
            session1.session_id, "مرحبا", "أهلا", 1000, transcription_quality="ok"
        )
        await session_manager.add_conversation_turn(
            session1.session_id, "كيف حالك؟", "بخير", 1500, transcription_quality="ok"
        )
        await session_manager.add_conversation_turn(
            session2.session_id, "unclear audio", "Please try again", 500, transcription_quality="unclear"
        )
        
        # Get statistics
        stats = session_manager.get_session_stats()
        
        assert stats["active_sessions"] == 2
        assert stats["total_turns"] == 3
        assert stats["average_turns_per_session"] == 1.5
        assert stats["average_response_time_ms"] > 0
        assert stats["average_transcription_quality"] < 1.0  # Due to one unclear transcription


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])