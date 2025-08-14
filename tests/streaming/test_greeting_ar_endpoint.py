#!/usr/bin/env python3
"""
Test script for the new streaming voice endpoint with Arabic greeting.
Sends greeting_ar.pcm and captures the streaming transcription output.
"""

import asyncio
import json
import time
import websockets
from pathlib import Path
import sys
from typing import Optional

# Expected reference transcript for validation
EXPECTED_TRANSCRIPT = "مرحبًا، كيف حالك اليوم؟ أتصل لأستفسر عن الخدمات المتوفرة في عيادة التجميل الخاصة بكم."

class GreetingTestSession:
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.pcm_file = Path("/home/lumi/beautyai/voice_tests/input_test_questions/pcm/greeting_ar.pcm")
        self.output_file = self.test_dir / "greeting_ar_test_output.json"
        self.ws_url = "ws://localhost:8000/api/v1/ws/streaming-voice?language=ar"
        
        # Results storage
        self.session_info = {}
        self.partial_transcripts = []
        self.final_transcripts = []
        self.connection_start = None
        self.first_response = None
        self.test_completed = False

    async def send_pcm_data(self, websocket):
        """Send PCM audio data to the streaming endpoint."""
        print(f"📁 Reading PCM file: {self.pcm_file}")
        
        if not self.pcm_file.exists():
            raise FileNotFoundError(f"PCM file not found: {self.pcm_file}")
        
        # Read the entire PCM file
        pcm_data = self.pcm_file.read_bytes()
        total_frames = len(pcm_data) // 2  # s16le = 2 bytes per sample
        print(f"🎵 PCM file size: {len(pcm_data):,} bytes ({total_frames:,} frames)")
        
        # Send PCM data in chunks (20ms frames at 16kHz = 320 samples = 640 bytes)
        frame_size = 640  # 20ms worth of data
        sent_frames = 0
        
        for i in range(0, len(pcm_data), frame_size):
            chunk = pcm_data[i:i + frame_size]
            await websocket.send(chunk)
            sent_frames += len(chunk) // 2
            
            # Real-time pacing: 20ms delay between frames
            await asyncio.sleep(0.02)
        
        print(f"✅ Sent {sent_frames:,} PCM frames ({len(pcm_data):,} bytes)")
        
        # Wait a bit for final processing
        await asyncio.sleep(2.0)

    async def receive_responses(self, websocket):
        """Receive and process streaming responses."""
        print("👂 Listening for streaming responses...")
        
        try:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_time = time.time()
                
                if self.first_response is None:
                    self.first_response = response_time
                    latency = (response_time - self.connection_start) * 1000
                    print(f"⚡ First response latency: {latency:.1f}ms")
                
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "ready":
                        self.session_info = data
                        print(f"🔗 Session ready: {data.get('session_id', 'N/A')[:8]}...")
                        print(f"🎯 Feature: {data.get('feature', 'N/A')}")
                        print(f"🌍 Language: {data.get('language', 'N/A')}")
                        
                    elif msg_type == "partial_transcript":
                        self.partial_transcripts.append(data)
                        text = data.get("text", "")
                        stable = data.get("stable", False)
                        mock = data.get("mock", False)
                        cursor = data.get("cursor", 0)
                        audio_level = data.get("audio_level", 0)
                        buffer_usage = data.get("buffer_usage", 0)
                        
                        status = "🔸 STABLE" if stable else "🔹 PARTIAL"
                        mock_indicator = " [MOCK]" if mock else ""
                        
                        print(f"{status} [{cursor}]{mock_indicator}: '{text[:60]}{'...' if len(text) > 60 else ''}' (level: {audio_level:.3f}, buffer: {buffer_usage:.1%})")
                        
                    elif msg_type == "final_transcript":
                        self.final_transcripts.append(data)
                        text = data.get("text", "")
                        utterance_id = data.get("utterance_id", "N/A")
                        duration_ms = data.get("duration_ms", 0)
                        total_frames = data.get("total_pcm_frames", 0)
                        mock = data.get("mock", False)
                        
                        mock_indicator = " [MOCK]" if mock else ""
                        print(f"🏁 FINAL{mock_indicator} [{utterance_id}]: '{text}' ({duration_ms}ms, {total_frames} frames)")
                        
                    else:
                        print(f"❓ Unknown message type: {msg_type}")
                        
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON: {message[:100]}...")
                    
        except asyncio.TimeoutError:
            print("⏰ Receive timeout - ending session")
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")

    async def run_test(self):
        """Run the complete test session."""
        print("🎤 Testing Arabic Greeting with Streaming Voice Endpoint")
        print("=" * 60)
        print(f"📁 PCM File: {self.pcm_file}")
        print(f"🔗 WebSocket URL: {self.ws_url}")
        print(f"📝 Expected: '{EXPECTED_TRANSCRIPT[:50]}...'")
        print()
        
        try:
            print("🔌 Connecting to WebSocket...")
            self.connection_start = time.time()
            
            async with websockets.connect(self.ws_url) as websocket:
                print("✅ WebSocket connected!")
                
                # Run sender and receiver concurrently
                await asyncio.gather(
                    self.send_pcm_data(websocket),
                    self.receive_responses(websocket)
                )
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        
        finally:
            await self.save_results()
            self.analyze_results()
            
        return True

    async def save_results(self):
        """Save test results to JSON file."""
        results = {
            "test_info": {
                "pcm_file": str(self.pcm_file),
                "expected_transcript": EXPECTED_TRANSCRIPT,
                "test_timestamp": time.time(),
                "websocket_url": self.ws_url
            },
            "session_info": self.session_info,
            "partial_transcripts": self.partial_transcripts,
            "final_transcripts": self.final_transcripts,
            "timing": {
                "connection_start": self.connection_start,
                "first_response": self.first_response,
                "first_response_latency_ms": (self.first_response - self.connection_start) * 1000 if self.first_response else None
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to: {self.output_file}")

    def analyze_results(self):
        """Analyze and compare results with expected transcript."""
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 60)
        
        # Session info
        if self.session_info:
            print(f"🔗 Session ID: {self.session_info.get('session_id', 'N/A')}")
            print(f"🎯 Feature: {self.session_info.get('feature', 'N/A')}")
            print(f"🌍 Language: {self.session_info.get('language', 'N/A')}")
            print(f"📊 Ring Buffer: {self.session_info.get('ring_buffer_seconds', 'N/A')}s")
        
        # Timing analysis
        if self.first_response and self.connection_start:
            latency = (self.first_response - self.connection_start) * 1000
            print(f"⚡ First Response Latency: {latency:.1f}ms")
        
        # Transcript analysis
        print(f"\n📝 Partial Transcripts: {len(self.partial_transcripts)}")
        print(f"🏁 Final Transcripts: {len(self.final_transcripts)}")
        
        if self.final_transcripts:
            # Get the most recent final transcript
            latest_final = self.final_transcripts[-1]
            actual_text = latest_final.get("text", "")
            is_mock = latest_final.get("mock", False)
            
            print(f"\n🎯 TRANSCRIPT COMPARISON:")
            print(f"📋 Expected: '{EXPECTED_TRANSCRIPT}'")
            print(f"📋 Actual  : '{actual_text}'")
            print(f"🤖 Mock Mode: {is_mock}")
            
            if is_mock:
                print("⚠️  Note: This is mock mode output, not real transcription")
                print("💡 To test real transcription, disable mock mode in the endpoint")
            else:
                # Simple comparison for real transcription
                if actual_text.strip() == EXPECTED_TRANSCRIPT.strip():
                    print("✅ PERFECT MATCH!")
                elif EXPECTED_TRANSCRIPT[:20] in actual_text or actual_text[:20] in EXPECTED_TRANSCRIPT:
                    print("🟡 PARTIAL MATCH - close enough")
                else:
                    print("❌ NO MATCH - transcription differs significantly")
        else:
            print("❌ No final transcripts received")
        
        # Audio processing stats
        if self.partial_transcripts:
            audio_levels = [t.get("audio_level", 0) for t in self.partial_transcripts if t.get("audio_level") is not None]
            buffer_usages = [t.get("buffer_usage", 0) for t in self.partial_transcripts if t.get("buffer_usage") is not None]
            
            if audio_levels:
                print(f"\n🔊 Audio Level: avg={sum(audio_levels)/len(audio_levels):.3f}, max={max(audio_levels):.3f}")
            if buffer_usages:
                print(f"📊 Buffer Usage: avg={sum(buffer_usages)/len(buffer_usages):.1%}, max={max(buffer_usages):.1%}")
        
        print(f"\n💾 Full results saved in: {self.output_file}")
        print("🎉 Test completed!")

async def main():
    """Main test function."""
    tester = GreetingTestSession()
    success = await tester.run_test()
    
    if success:
        print("\n✅ Test completed successfully!")
        return 0
    else:
        print("\n❌ Test failed!")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
