#!/usr/bin/env python3
"""
Focused Voice-to-Voice Debug Script.
WebM input only, with comprehensive input/output transcription and audio saving.
Designed to identify root causes of persistent issues.
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

class VoiceDebugTester:
    def __init__(self):
        self.api_url = "http://localhost:8000/inference/voice-to-voice"
        self.test_file = "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.webm"
        self.output_dir = Path("/home/lumi/beautyai/voice_tests/debug_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize audio transcription service
        self.audio_service = AudioTranscriptionService()
        
    def transcribe_audio_direct(self, audio_path, model_name="whisper-large-v3-turbo-arabic"):
        """Directly transcribe audio file using the AudioTranscriptionService."""
        print(f"🎯 Transcribing audio directly: {audio_path}")
        
        try:
            # Load the Whisper model
            print(f"📥 Loading Whisper model: {model_name}")
            if not self.audio_service.load_whisper_model(model_name):
                print(f"❌ Failed to load Whisper model: {model_name}")
                return None, 0
            
            # Transcribe the audio file
            print(f"🎤 Processing audio file...")
            transcription = self.audio_service.transcribe_audio_file(audio_path, language="ar")
            
            if transcription:
                print(f"✅ Direct transcription: {transcription}")
                # For now, we'll set confidence to 1.0 since the service doesn't return it
                confidence = 1.0
                return transcription, confidence
            else:
                print(f"❌ Transcription failed")
                return None, 0
                    
        except Exception as e:
            print(f"❌ Exception during direct transcription: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def save_output_audio(self, audio_data, filename_prefix):
        """Save output audio data to file."""
        if not audio_data:
            print("❌ No audio data to save")
            return None
            
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Save as WAV file
            output_path = self.output_dir / f"{filename_prefix}_{self.timestamp}.wav"
            
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
                
            print(f"💾 Output audio saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            return None
    
    def analyze_response_text(self, response_text):
        """Analyze response text for thinking content and other issues."""
        print(f"\n🔍 RESPONSE TEXT ANALYSIS")
        print("-" * 50)
        
        # Check for thinking content
        thinking_patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'\*thinks\*',
            r'\*thinking\*',
            r'Let me think',
            r'I need to think'
        ]
        
        issues_found = []
        
        # Check for thinking content
        if '<think>' in response_text or '</think>' in response_text:
            issues_found.append("Contains <think> tags")
            
        if '<thinking>' in response_text or '</thinking>' in response_text:
            issues_found.append("Contains <thinking> tags")
            
        # Check for non-Arabic content (basic check)
        arabic_chars = sum(1 for char in response_text if '\u0600' <= char <= '\u06FF')
        total_chars = len(response_text.replace(' ', '').replace('\n', ''))
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
        
        print(f"📝 Response length: {len(response_text)} characters")
        print(f"🔤 Arabic character ratio: {arabic_ratio:.2f}")
        print(f"🧹 Issues found: {issues_found if issues_found else 'None'}")
        
        if issues_found:
            print(f"❌ PROBLEMS DETECTED: {', '.join(issues_found)}")
        else:
            print("✅ Response appears clean")
            
        return issues_found
    
    def run_comprehensive_test(self):
        """Run comprehensive test with input/output transcription."""
        
        print("🎤 FOCUSED VOICE-TO-VOICE DEBUG TEST")
        print("=" * 60)
        print(f"📁 Test file: {self.test_file}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏰ Test timestamp: {self.timestamp}")
        
        # Check if test file exists
        if not Path(self.test_file).exists():
            print(f"❌ Test file not found: {self.test_file}")
            return
        
        # Step 1: Direct transcription of input
        print(f"\n" + "="*60)
        print("📋 STEP 1: DIRECT INPUT TRANSCRIPTION")
        print("="*60)
        
        input_transcription, input_confidence = self.transcribe_audio_direct(self.test_file)
        
        if not input_transcription:
            print("❌ Failed to transcribe input - aborting test")
            return
            
        # Step 2: Voice-to-Voice API call
        print(f"\n" + "="*60)
        print("🚀 STEP 2: VOICE-TO-VOICE API CALL")
        print("="*60)
        
        test_data = {
            "input_language": "ar",
            "output_language": "ar",
            "stt_model_name": "whisper-large-v3-turbo-arabic",
            "tts_model_name": "coqui-tts-arabic",
            "chat_model_name": "qwen3-unsloth-q4ks",
            "speaker_voice": "female",
            "preset": "balanced",
            "disable_content_filter": "true",
            "thinking_mode": "false"  # Explicitly disable thinking mode
        }
        
        try:
            with open(self.test_file, "rb") as audio_file:
                files = {"audio_file": audio_file}
                
                print("🚀 Calling voice-to-voice API...")
                response = requests.post(self.api_url, files=files, data=test_data, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print("✅ API call successful!")
                    print(f"📝 API Transcription: {result.get('transcription', 'N/A')}")
                    print(f"💬 Response Text: {result.get('response_text', 'N/A')}")
                    print(f"⏱️ Processing Time: {result.get('total_processing_time_ms', 0):.1f}ms")
                    
                    # Step 3: Analyze response text
                    print(f"\n" + "="*60)
                    print("🔍 STEP 3: RESPONSE TEXT ANALYSIS")
                    print("="*60)
                    
                    response_text = result.get('response_text', '')
                    issues = self.analyze_response_text(response_text)
                    
                    # Step 4: Save and transcribe output audio
                    print(f"\n" + "="*60)
                    print("💾 STEP 4: OUTPUT AUDIO PROCESSING")
                    print("="*60)
                    
                    # Get audio data
                    audio_data = result.get('data', {}).get('audio_output_base64')
                    
                    if audio_data:
                        # Save output audio
                        output_audio_path = self.save_output_audio(audio_data, "voice_output")
                        
                        if output_audio_path:
                            # Transcribe output audio
                            print("\n🎯 Transcribing output audio...")
                            output_transcription, output_confidence = self.transcribe_audio_direct(output_audio_path)
                            
                            # Step 5: Comparison analysis
                            print(f"\n" + "="*60)
                            print("📊 STEP 5: COMPARISON ANALYSIS")
                            print("="*60)
                            
                            print(f"🎤 INPUT TRANSCRIPTION:")
                            print(f"   Text: {input_transcription}")
                            print(f"   Confidence: {input_confidence:.2f}")
                            
                            print(f"\n🤖 API TRANSCRIPTION:")
                            print(f"   Text: {result.get('transcription', 'N/A')}")
                            
                            print(f"\n💬 LLM RESPONSE TEXT:")
                            print(f"   Text: {response_text}")
                            print(f"   Length: {len(response_text)} chars")
                            print(f"   Issues: {issues if issues else 'None'}")
                            
                            if output_transcription:
                                print(f"\n🔊 OUTPUT AUDIO TRANSCRIPTION:")
                                print(f"   Text: {output_transcription}")
                                print(f"   Confidence: {output_confidence:.2f}")
                                
                                # Check if output matches response text (conceptually)
                                if response_text and output_transcription:
                                    # Simple similarity check
                                    response_words = set(response_text.split())
                                    output_words = set(output_transcription.split())
                                    if response_words and output_words:
                                        similarity = len(response_words & output_words) / len(response_words | output_words)
                                        print(f"📈 Text-to-Audio similarity: {similarity:.2f}")
                                        
                                        if similarity < 0.3:
                                            print("⚠️ LOW SIMILARITY: Output audio may not match response text")
                                        else:
                                            print("✅ Good similarity between response text and output audio")
                    else:
                        print("❌ No audio data in API response")
                    
                    # Step 6: Summary
                    print(f"\n" + "="*60)
                    print("🏁 FINAL SUMMARY")
                    print("="*60)
                    
                    if issues:
                        print("❌ ISSUES DETECTED:")
                        for issue in issues:
                            print(f"   • {issue}")
                    else:
                        print("✅ NO OBVIOUS ISSUES DETECTED")
                        
                    # Transcription consistency check
                    api_trans = result.get('transcription', '')
                    if input_transcription and api_trans:
                        if input_transcription.strip() == api_trans.strip():
                            print("✅ Input transcription consistency: PERFECT MATCH")
                        else:
                            print("⚠️ Input transcription consistency: MISMATCH")
                            print(f"   Direct: {input_transcription}")
                            print(f"   API:    {api_trans}")
                            
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
    tester = VoiceDebugTester()
    tester.run_comprehensive_test()
