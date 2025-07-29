#!/usr/bin/env python3
"""
Final Voice-to-Voice Debug Script with Arabic Model.
Tests with Arabic-specific model to solve language response issue.
"""
import requests
import json
import base64
from pathlib import Path
from datetime import datetime
import tempfile
import subprocess
import os
import sys

# Add the beautyai_inference module to the path
sys.path.insert(0, '/home/lumi/beautyai')

# Import the audio transcription service directly
from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService

class FinalVoiceDebugTester:
    def __init__(self):
        self.api_url = "http://localhost:8000/inference/voice-to-voice"
        self.test_file = "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.webm"
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/debug_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize audio transcription service
        self.audio_service = AudioTranscriptionService()
        
    def test_with_arabic_model(self):
        """Test voice-to-voice with Arabic-specific model."""
        
        print("🎤 FINAL VOICE-TO-VOICE TEST WITH ARABIC MODEL")
        print("=" * 60)
        print(f"📁 Test file: {self.test_file}")
        print(f"⏰ Test timestamp: {self.timestamp}")
        
        # Check if test file exists
        if not Path(self.test_file).exists():
            print(f"❌ Test file not found: {self.test_file}")
            return
        
        # Test with Arabic-specific model
        print(f"\n" + "="*60)
        print("🚀 TESTING WITH ARABIC REASONING MODEL")
        print("="*60)
        
        test_data = {
            "input_language": "ar",
            "output_language": "ar",
            "stt_model_name": "whisper-large-v3-turbo-arabic",
            "tts_model_name": "coqui-tts-arabic",
            "chat_model_name": "bee1reason-arabic-q4ks",  # Arabic-specific model
            "speaker_voice": "female",
            "preset": "balanced",
            "disable_content_filter": "true",
            "thinking_mode": "false"
        }
        
        try:
            with open(self.test_file, "rb") as audio_file:
                files = {"audio_file": audio_file}
                
                print("🚀 Calling voice-to-voice API with Arabic model...")
                response = requests.post(self.api_url, files=files, data=test_data, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print("✅ API call successful!")
                    print(f"📝 Transcription: {result.get('transcription', 'N/A')}")
                    
                    response_text = result.get('response_text', '')
                    print(f"💬 Response Text: {response_text}")
                    print(f"⏱️ Processing Time: {result.get('total_processing_time_ms', 0):.1f}ms")
                    
                    # Check for Arabic content
                    arabic_chars = sum(1 for char in response_text if '\u0600' <= char <= '\u06FF')
                    total_chars = len(response_text.replace(' ', '').replace('\n', ''))
                    arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
                    
                    print(f"🔤 Arabic character ratio: {arabic_ratio:.2f}")
                    
                    if arabic_ratio > 0.7:
                        print("✅ SUCCESS: Model is responding in Arabic!")
                    elif arabic_ratio > 0.3:
                        print("⚠️ PARTIAL: Model is responding with mixed Arabic/English")
                    else:
                        print("❌ FAILED: Model is still responding in English")
                    
                    # Check for audio
                    audio_data = result.get('data', {}).get('audio_output_base64')
                    if audio_data:
                        print("✅ Audio data received in API response")
                        
                        # Save and test audio
                        output_path = self.output_dir / f"arabic_model_output_{self.timestamp}.wav"
                        audio_bytes = base64.b64decode(audio_data)
                        
                        with open(output_path, "wb") as f:
                            f.write(audio_bytes)
                        
                        print(f"💾 Audio saved: {output_path}")
                        
                        # Quick transcription test
                        print("🎯 Testing output audio transcription...")
                        if self.audio_service.load_whisper_model("whisper-large-v3-turbo-arabic"):
                            output_transcription = self.audio_service.transcribe_audio_file(str(output_path), language="ar")
                            if output_transcription:
                                print(f"🔊 Output transcription: {output_transcription[:100]}...")
                                
                                # Check if output is Arabic
                                output_arabic_chars = sum(1 for char in output_transcription if '\u0600' <= char <= '\u06FF')
                                output_total_chars = len(output_transcription.replace(' ', '').replace('\n', ''))
                                output_arabic_ratio = output_arabic_chars / output_total_chars if output_total_chars > 0 else 0
                                
                                print(f"🔊 Output audio Arabic ratio: {output_arabic_ratio:.2f}")
                                
                                if output_arabic_ratio > 0.7:
                                    print("✅ SUCCESS: TTS output is in Arabic!")
                                    print("🎉 COMPLETE PIPELINE SUCCESS!")
                                else:
                                    print("❌ TTS output is not in Arabic")
                            
                    else:
                        print("❌ No audio data in API response")
                    
                else:
                    print(f"❌ API call failed: {response.status_code}")
                    try:
                        error = response.json()
                        print(f"Error details: {error}")
                    except:
                        print(f"Raw error: {response.text}")
                        
        except Exception as e:
            print(f"❌ Exception during API call: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tester = FinalVoiceDebugTester()
    tester.test_with_arabic_model()
