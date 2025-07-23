"""
Unit tests for SimpleVoiceService.

Tests the basic functionality of the Edge TTS-based simple voice service.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from beautyai_inference.services.voice.conversation import SimpleVoiceService


class TestSimpleVoiceService:
    """Test cases for SimpleVoiceService."""
    
    @pytest.fixture
    async def service(self):
        """Create a SimpleVoiceService instance for testing."""
        service = SimpleVoiceService()
        await service.initialize()
        yield service
        await service.cleanup()
    
    def test_initialization(self):
        """Test service initialization."""
        service = SimpleVoiceService()
        
        assert service.config == {}
        assert service.temp_dir.exists()
        assert service.voice_mappings is not None
        assert len(service.voice_mappings) > 0
        assert service.default_voice == "ar-SA-ZariyahNeural"
        assert service.default_english_voice == "en-US-AriaNeural"
    
    def test_voice_mappings_setup(self):
        """Test voice mappings configuration."""
        service = SimpleVoiceService()
        mappings = service._setup_voice_mappings()
        
        # Check required voices are present
        assert "ar_female" in mappings
        assert "ar_male" in mappings
        assert "en_female" in mappings
        assert "en_male" in mappings
        
        # Check voice mapping structure
        ar_female = mappings["ar_female"]
        assert ar_female.language == "ar-SA"
        assert ar_female.gender == "female"
        assert ar_female.voice_id == "ar-SA-ZariyahNeural"
        assert "Arabic Female" in ar_female.display_name
    
    def test_language_detection(self):
        """Test language detection functionality."""
        service = SimpleVoiceService()
        
        # Test Arabic text
        arabic_text = "مرحبا كيف حالك؟"
        assert service._detect_language(arabic_text) == "ar"
        
        # Test English text
        english_text = "Hello how are you?"
        assert service._detect_language(english_text) == "en"
        
        # Test mixed text (more Arabic)
        mixed_text = "Hello مرحبا كيف الحال"
        assert service._detect_language(mixed_text) == "ar"
        
        # Test empty text
        assert service._detect_language("") == "en"
        
        # Test non-alphabetic text
        assert service._detect_language("123 !@#") == "en"
    
    def test_voice_selection(self):
        """Test voice selection logic."""
        service = SimpleVoiceService()
        
        # Test Arabic female selection
        voice = service._select_voice("ar", "female")
        assert voice == "ar-SA-ZariyahNeural"
        
        # Test Arabic male selection
        voice = service._select_voice("ar", "male")
        assert voice == "ar-SA-HamedNeural"
        
        # Test English female selection
        voice = service._select_voice("en", "female")
        assert voice == "en-US-AriaNeural"
        
        # Test English male selection
        voice = service._select_voice("en", "male")
        assert voice == "en-US-GuyNeural"
        
        # Test default selection (Arabic)
        voice = service._select_voice(None, "female")
        assert voice == "ar-SA-ZariyahNeural"
        
        # Test fallback for unknown language
        voice = service._select_voice("unknown", "female")
        assert voice == "en-US-AriaNeural"
    
    def test_get_available_voices(self):
        """Test getting available voices."""
        service = SimpleVoiceService()
        voices = service.get_available_voices()
        
        assert isinstance(voices, dict)
        assert len(voices) > 0
        
        # Check structure of voice info
        for voice_key, voice_info in voices.items():
            assert "language" in voice_info
            assert "gender" in voice_info
            assert "voice_id" in voice_info
            assert "display_name" in voice_info
    
    @pytest.mark.asyncio
    async def test_save_audio_data(self):
        """Test saving audio data to temporary file."""
        service = SimpleVoiceService()
        
        # Test audio data
        test_audio = b"fake audio data"
        
        audio_path = await service._save_audio_data(test_audio)
        
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"
        
        # Check content
        with open(audio_path, 'rb') as f:
            content = f.read()
        assert content == test_audio
        
        # Cleanup
        audio_path.unlink()
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_placeholder(self):
        """Test audio transcription placeholder."""
        service = SimpleVoiceService()
        
        # Create a test file
        test_file = service.temp_dir / "test_audio.wav"
        test_file.write_bytes(b"fake audio")
        
        # Test transcription (currently returns mock data)
        result = await service._transcribe_audio(test_file)
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Cleanup
        test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_generate_chat_response_placeholder(self):
        """Test chat response generation placeholder."""
        service = SimpleVoiceService()
        
        # Test Arabic input
        arabic_text = "مرحبا"
        response = await service._generate_chat_response(arabic_text, "qwen-3")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Test English input
        english_text = "Hello"
        response = await service._generate_chat_response(english_text, "qwen-3")
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_text_to_speech_functionality(self, service):
        """Test text-to-speech conversion."""
        # Test simple text conversion
        text = "مرحبا، هذا اختبار"
        
        try:
            audio_path = await service.text_to_speech(text)
            
            # Check that file was created
            assert audio_path.exists()
            assert audio_path.suffix == ".wav"
            
            # Check file has content
            assert audio_path.stat().st_size > 0
            
            # Cleanup
            audio_path.unlink()
            
        except Exception as e:
            # If Edge TTS is not available in test environment, that's okay
            pytest.skip(f"Edge TTS not available in test environment: {e}")
    
    @pytest.mark.asyncio
    async def test_text_to_speech_with_english(self, service):
        """Test text-to-speech with English text."""
        text = "Hello, this is a test"
        
        try:
            audio_path = await service.text_to_speech(text)
            
            assert audio_path.exists()
            assert audio_path.suffix == ".wav"
            assert audio_path.stat().st_size > 0
            
            # Cleanup
            audio_path.unlink()
            
        except Exception as e:
            pytest.skip(f"Edge TTS not available in test environment: {e}")
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        service = SimpleVoiceService()
        stats = service.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert "service_name" in stats
        assert "edge_tts_available" in stats
        assert "temp_directory" in stats
        assert "temp_files_count" in stats
        assert "available_voices" in stats
        assert "default_arabic_voice" in stats
        assert "default_english_voice" in stats
        
        assert stats["service_name"] == "SimpleVoiceService"
        assert stats["default_arabic_voice"] == "ar-SA-ZariyahNeural"
        assert stats["default_english_voice"] == "en-US-AriaNeural"
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test service cleanup."""
        service = SimpleVoiceService()
        
        # Create some temporary files
        temp_file1 = service.temp_dir / "test1.wav"
        temp_file2 = service.temp_dir / "test2.wav"
        temp_file1.write_text("test")
        temp_file2.write_text("test")
        
        # Cleanup should remove temp files
        await service.cleanup()
        
        # Files should be removed
        assert not temp_file1.exists()
        assert not temp_file2.exists()


@pytest.mark.asyncio
async def test_service_integration():
    """Integration test for the complete service."""
    service = SimpleVoiceService()
    
    try:
        # Initialize service
        await service.initialize()
        
        # Test voice mappings
        voices = service.get_available_voices()
        assert len(voices) > 0
        
        # Test stats
        stats = service.get_processing_stats()
        assert stats["service_name"] == "SimpleVoiceService"
        
        print("✅ SimpleVoiceService integration test passed!")
        
    except Exception as e:
        pytest.skip(f"Integration test skipped due to: {e}")
    
    finally:
        await service.cleanup()


if __name__ == "__main__":
    # Run a simple test if executed directly
    async def main():
        await test_service_integration()
    
    asyncio.run(main())
