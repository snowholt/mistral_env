#!/usr/bin/env python3
"""
WebSocket Simple Voice Integration Test with OpenAI Validation

This script tests the complete pipeline:
1. Connect to WebSocket Simple Voice endpoint
2. Upload audio file (q7.webm)
3. Receive transcription + LLM response + TTS audio
4. Send TTS audio back to OpenAI for transcription
5. Compare LLM response with OpenAI transcription of TTS

Purpose: Identify issues in the voice processing pipeline by comparing
against OpenAI's reference implementation.

Author: BeautyAI Framework
Date: 2025-10-06
"""

import asyncio
import json
import time
import base64
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import websockets

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import os
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install openai python-dotenv websockets")
    sys.exit(1)


class WebSocketVoiceIntegrationTest:
    """Integration test for voice pipeline with OpenAI quality validation."""
    
    def __init__(self, audio_file: str = "q7.webm", results_file: str = "integration_test_results.json"):
        """
        Initialize the integration test.
        
        Args:
            audio_file: Path to audio file to test
            results_file: Path to save results
        """
        self.script_dir = Path(__file__).parent
        self.audio_file = self.script_dir / audio_file
        self.results_file = self.script_dir / results_file
        
        # WebSocket configuration (with /api/v1 prefix)
        self.ws_url = "ws://localhost:8000/api/v1/ws/simple-voice-chat"
        self.language = "ar"
        self.voice_type = "female"
        
        # OpenAI client (will be set externally)
        self.openai_client = None
        
        # Results storage
        self.results = {}
    
    async def connect_websocket(self, url: str = "ws://localhost:8000/ws/simple-voice-chat",
                                language: str = "ar", voice_type: str = "female",
                                debug_mode: bool = True) -> bool:
        """
        Connect to WebSocket Simple Voice endpoint.
        
        Args:
            url: WebSocket URL
            language: Language code (ar/en)
            voice_type: Voice type (male/female)
            debug_mode: Enable debug mode
            
        Returns:
            True if connection successful
        """
        print("\n🔌 Connecting to WebSocket...")
        print(f"   URL: {url}")
        print(f"   Language: {language}")
        print(f"   Voice: {voice_type}")
        print(f"   Debug: {debug_mode}")
        
        try:
            full_url = f"{url}?language={language}&voice_type={voice_type}&debug_mode={debug_mode}"
            self.websocket = await websockets.connect(full_url)
            
            print(f"✅ Connected successfully")
            self.results["connection"] = {
                "success": True,
                "url": full_url,
                "timestamp": time.time()
            }
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.results["connection"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    async def upload_audio_file(self, audio_path: Path) -> bool:
        """
        Upload audio file to WebSocket.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if upload successful
        """
        print(f"\n📤 Uploading audio file: {audio_path.name}")
        print(f"   File size: {audio_path.stat().st_size / 1024:.2f} KB")
        
        if not self.websocket:
            print(f"❌ Not connected to WebSocket")
            return False
        
        try:
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Send as binary message
            await self.websocket.send(audio_data)
            
            print(f"✅ Audio uploaded ({len(audio_data)} bytes)")
            self.results["upload"] = {
                "success": True,
                "file": str(audio_path),
                "size_bytes": len(audio_data),
                "timestamp": time.time()
            }
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            self.results["upload"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    async def receive_response(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Receive response from WebSocket.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Response dictionary or None
        """
        print(f"\n⏳ Waiting for response (timeout: {timeout}s)...")
        
        if not self.websocket:
            print(f"❌ Not connected to WebSocket")
            return None
        
        try:
            # Wait for voice_response message
            start_time = time.time()
            while time.time() - start_time < timeout:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=timeout - (time.time() - start_time)
                )
                
                # Parse JSON message
                data = json.loads(message)
                
                # Log all message types
                msg_type = data.get("type", "unknown")
                print(f"   📨 Received: {msg_type}")
                
                # Wait for voice_response
                if msg_type == "voice_response":
                    elapsed_time = time.time() - start_time
                    print(f"✅ Voice response received ({elapsed_time:.2f}s)")
                    
                    # Extract data
                    transcription = data.get("transcription", "")
                    response_text = data.get("response_text", "")
                    audio_base64 = data.get("audio_base64", "")
                    
                    print(f"\n📝 Transcription: {transcription}")
                    print(f"💬 Response: {response_text[:150]}{'...' if len(response_text) > 150 else ''}")
                    print(f"🔊 Audio: {len(audio_base64)} base64 chars")
                    
                    # Store results
                    self.results["transcription"] = {
                        "success": True,
                        "text": transcription,
                        "timestamp": time.time(),
                        "processing_time_sec": elapsed_time
                    }
                    
                    self.results["llm_response"] = {
                        "success": True,
                        "text": response_text,
                        "timestamp": time.time()
                    }
                    
                    self.results["tts_audio"] = {
                        "success": True,
                        "base64": audio_base64,
                        "size_bytes": len(audio_base64) * 3 // 4,  # Approximate
                        "timestamp": time.time()
                    }
                    
                    return data
            
            print(f"⏱️  Timeout waiting for response")
            return None
            
        except asyncio.TimeoutError:
            print(f"⏱️  Timeout waiting for response")
            return None
        except Exception as e:
            print(f"❌ Error receiving response: {e}")
            return None
    
    async def transcribe_tts_audio(self, audio_base64: str) -> Optional[Dict[str, Any]]:
        """
        Transcribe TTS audio using OpenAI Whisper.
        
        Args:
            audio_base64: Base64-encoded TTS audio
            
        Returns:
            Transcription result or None
        """
        print(f"\n🎤 Transcribing TTS audio with OpenAI Whisper...")
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            # Transcribe with OpenAI
            with open(temp_path, 'rb') as f:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="ar"
                )
            
            # Cleanup
            Path(temp_path).unlink()
            
            tts_text = transcription.text
            print(f"✅ TTS Transcription: {tts_text}")
            
            self.results["tts_transcription"] = {
                "success": True,
                "text": tts_text,
                "timestamp": time.time()
            }
            
            return {"text": tts_text}
            
        except Exception as e:
            print(f"❌ TTS transcription failed: {e}")
            self.results["tts_transcription"] = {
                "success": False,
                "error": str(e)
            }
            return None
    
    def compare_responses(self, expected: str, actual: str) -> Dict[str, Any]:
        """
        Compare expected LLM response text with actual TTS transcription.
        
        Args:
            expected: Expected text (LLM response)
            actual: Actual text (TTS transcription)
            
        Returns:
            Comparison results with similarity score
        """
        print(f"\n📊 Comparing LLM response with TTS transcription...")
        
        if not expected or not actual:
            print(f"❌ Missing data for comparison")
            return {"success": False, "error": "Missing data"}
        
        # Simple similarity check (character-level)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, expected, actual).ratio()
        
        # Determine quality level
        if similarity > 0.95:
            quality = "excellent"
            note = "TTS quality is excellent"
        elif similarity > 0.80:
            quality = "good"
            note = "TTS quality is good"
        elif similarity > 0.70:
            quality = "fair"
            note = "TTS quality is fair"
        else:
            quality = "poor"
            note = "Low similarity - potential TTS quality issue"
        
        result = {
            "success": True,
            "similarity_score": similarity,
            "quality_level": quality,
            "note": note,
            "expected": expected,
            "actual": actual,
            "expected_length": len(expected),
            "actual_length": len(actual)
        }
        
        self.results["comparison"] = result
        
        return result
    
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run complete integration test pipeline.
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print("🚀 WebSocket Voice Integration Test with OpenAI Validation")
        print("="*70)
        
        # Use the audio file path from initialization
        audio_path = self.audio_file
        
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        if self.openai_client is None:
            print(f"❌ OpenAI client not initialized")
            return None
        
        # Step 1: Connect to WebSocket
        print(f"\n🔌 Connecting to WebSocket...")
        print(f"   URL: {self.ws_url}")
        print(f"   Language: {self.language}")
        print(f"   Voice: {self.voice_type}")
        print(f"   Debug: True")
        
        connected = await self.connect_websocket(
            url=self.ws_url,
            language=self.language,
            voice_type=self.voice_type,
            debug_mode=True
        )
        
        if not connected:
            print("❌ WebSocket connection failed")
            return self.results
        
        # Step 2: Upload audio
        print(f"\n📤 Uploading audio file: {audio_path.name}")
        upload_success = await self.upload_audio_file(audio_path)
        
        if not upload_success:
            print("❌ Audio upload failed")
            return self.results
        
        # Step 3: Receive response
        print(f"\n⏳ Waiting for voice response...")
        response_data = await self.receive_response()
        
        if not response_data:
            print("❌ No response received")
            return self.results
        
        # Step 4: Transcribe TTS audio with OpenAI
        if response_data.get("audio_base64"):
            tts_result = await self.transcribe_tts_audio(response_data["audio_base64"])
            
            # Step 5: Compare responses
            if tts_result:
                comparison = self.compare_responses(
                    expected=response_data.get("response_text", ""),
                    actual=tts_result.get("text", "")
                )
                
                print(f"\n📊 Quality Assessment:")
                print(f"   Similarity Score: {comparison['similarity_score']:.2%}")
                print(f"   Quality: {comparison['quality_level']}")
                if comparison.get("note"):
                    print(f"   Note: {comparison['note']}")
        
        # Final summary
        print("\n" + "="*70)
        if self.results.get("connection", {}).get("success"):
            print("✅ Integration test completed successfully")
            self.results["success"] = True
        else:
            print("❌ Integration test failed")
            self.results["success"] = False
        print("="*70)
        
        return self.results
    
    def save_results(self, output_path: Optional[Path] = None):
        """Save test results to JSON file."""
        if output_path is None:
            output_path = Path(__file__).parent / "integration_test_results.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


async def main():
    """Main execution function."""
    # Load OpenAI credentials
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    # Configuration
    audio_file = Path(__file__).parent / "q7.webm"
    
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=api_key)
        
        # Run integration test (pass only audio filename, not full path)
        tester = WebSocketVoiceIntegrationTest(audio_file="q7.webm", results_file="integration_test_results.json")
        tester.openai_client = openai_client  # Add OpenAI client
        
        results = await tester.run_full_pipeline()
        
        # Save results
        tester.save_results()
        
        # Exit with appropriate code
        if results is None:
            print("❌ Test returned no results")
            sys.exit(1)
        sys.exit(0 if results.get('success') else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
