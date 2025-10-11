#!/usr/bin/env python3
"""
Batch WebSocket Voice Testing Script

Runs comprehensive tests on all validated voice files (WebM format)
through the complete voice pipeline: STT → LLM → TTS → Validation
"""

import asyncio
import websockets
import json
import time
import sys
import base64
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os
from difflib import SequenceMatcher

# Expected transcriptions for numbered files
EXPECTED_TRANSCRIPTIONS = {
    "1": "ما هي تكلفة حقن البوتوكس؟",
    "2": "كم جلسة ليزر أحتاج لإزالة الشعر؟",
    "3": "هل زراعة الأسنان مؤلمة؟",
    "4": "متى تظهر نتائج عملية تجميل الأنف؟",
    "5": "ما هي مدة التعافي بعد حقن الفيلر؟",
    "6": "هل الليزر آمن للبشرة الحساسة؟",
    "7": "كم سعر زراعة سن واحد؟",
    "8": "ما الفرق بين البوتوكس والفيلر؟",
    "9": "هل يمكن إزالة التاتو بالليزر؟",
    "10": "متى أستطيع العودة للعمل بعد عملية الأنف؟",
    "11": "كم تستغرق جلسة تبييض الأسنان؟",
    "12": "ما هي أنواع الفيلر المتاحة؟",
    "13": "هل علاج حب الشباب بالليزر فعال؟",
    "14": "ما هي مخاطر عملية شفط الدهون؟",
    "15": "كم مدة صلاحية حقن البوتوكس؟",
    "16": "هل يمكن عمل تقويم أسنان للكبار؟",
    "17": "ما تكلفة عملية شد الوجه؟",
    "18": "هل الليزر يزيل آثار الحروق؟",
    "19": "كم يستمر تأثير الفيلر تحت العين؟",
    "20": "ما هي أفضل طريقة لتكبير الشفاه؟",
    "21": "هل زراعة الشعر نتائجها دائمة؟",
    "22": "كم جلسة ليزر للتخلص من البقع؟",
    "23": "ما الفرق بين التقشير الكيميائي والليزر؟",
    "24": "هل يمكن حقن البوتوكس أثناء الحمل؟",
    "25": "ما تكلفة ابتسامة هوليود الكاملة؟",
    "26": "كم مدة الشفاء بعد عملية تجميل الجفون؟",
    "27": "هل الليزر يعالج الندبات القديمة؟",
    "28": "ما هي موانع استخدام الفيلر؟",
    "29": "كم سعر جلسة البلازما للوجه؟",
    "30": "هل يمكن الجمع بين البوتوكس والفيلر؟"
}


class BatchWebSocketTester:
    """Batch test WebSocket voice pipeline with multiple audio files."""
    
    def __init__(self, ws_url: str, voice_dir: Path, api_key: str):
        """
        Initialize batch tester.
        
        Args:
            ws_url: WebSocket URL
            voice_dir: Directory containing voice files
            api_key: OpenAI API key for validation
        """
        self.ws_url = ws_url
        self.voice_dir = voice_dir
        self.openai_client = OpenAI(api_key=api_key)
        self.results = []
        self.summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "total_time": 0,
            "avg_processing_time": 0,
            "avg_quality_score": 0
        }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    async def test_single_file(self, audio_file: Path, test_num: int, total: int) -> Dict:
        """
        Test a single audio file through the WebSocket pipeline.
        
        Args:
            audio_file: Path to audio file
            test_num: Current test number
            total: Total number of tests
            
        Returns:
            Test result dictionary
        """
        print(f"\n{'='*70}")
        print(f"🧪 Test {test_num}/{total}: {audio_file.name}")
        print(f"{'='*70}")
        
        result = {
            "test_number": test_num,
            "file": audio_file.name,
            "file_size_kb": audio_file.stat().st_size / 1024,
            "timestamp": time.time(),
            "success": False
        }
        
        try:
            # Connect to WebSocket
            print(f"  🔌 Connecting to WebSocket...")
            start_connect = time.time()
            
            async with websockets.connect(self.ws_url) as websocket:
                connect_time = time.time() - start_connect
                result["connect_time"] = connect_time
                print(f"  ✅ Connected in {connect_time:.3f}s")
                
                # Upload audio file
                print(f"  📤 Uploading audio ({result['file_size_kb']:.1f} KB)...")
                start_upload = time.time()
                
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                await websocket.send(audio_data)
                upload_time = time.time() - start_upload
                result["upload_time"] = upload_time
                print(f"  ✅ Uploaded in {upload_time:.3f}s")
                
                # Receive responses (connection → processing → voice_response)
                print(f"  ⏳ Waiting for voice response...")
                start_response = time.time()
                
                voice_response_data = None
                responses_received = 0
                
                try:
                    while responses_received < 5:  # Safety limit
                        response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        responses_received += 1
                        
                        try:
                            data = json.loads(response)
                            
                            if data.get("type") == "connection_established":
                                print(f"  📡 Connection established ({responses_received}/~3)")
                            elif data.get("type") == "processing_started":
                                print(f"  ⚙️  Processing started ({responses_received}/~3)")
                            elif data.get("type") == "voice_response":
                                voice_response_data = data
                                print(f"  🎯 Voice response received ({responses_received}/~3)")
                                break
                            else:
                                print(f"  ℹ️  Unknown message type: {data.get('type')}")
                        except json.JSONDecodeError:
                            # Binary data - shouldn't happen, but handle it
                            print(f"  ⚠️  Unexpected binary response")
                            continue
                    
                    response_time = time.time() - start_response
                    result["response_time"] = response_time
                    
                    if not voice_response_data:
                        result["error"] = f"No voice_response after {responses_received} messages"
                        result["success"] = False
                        print(f"  ❌ No voice_response received!")
                        return result
                    
                    # Extract base64 audio from voice_response
                    audio_base64 = voice_response_data.get("audio_base64")
                    if not audio_base64:
                        result["error"] = "No audio_base64 in voice_response"
                        result["success"] = False
                        print(f"  ❌ No audio data in response!")
                        return result
                    
                    # Decode base64 audio
                    tts_audio_bytes = base64.b64decode(audio_base64)
                    result["response_type"] = "audio"
                    result["response_size_kb"] = len(tts_audio_bytes) / 1024
                    
                    # Save TTS audio temporarily
                    temp_audio = Path(f"/tmp/tts_output_{test_num}.webm")
                    with open(temp_audio, 'wb') as f:
                        f.write(tts_audio_bytes)
                    
                    # Validate TTS quality with OpenAI
                    print(f"  🎤 Validating TTS quality...")
                    start_validation = time.time()
                    
                    with open(temp_audio, 'rb') as f:
                        transcription = self.openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=f,
                            language="ar"
                        )
                    
                    validation_time = time.time() - start_validation
                    result["validation_time"] = validation_time
                    result["tts_transcription"] = transcription.text
                    result["input_transcription"] = voice_response_data.get("transcription", "")
                    result["llm_response"] = voice_response_data.get("response_text", "")
                    
                    # Calculate quality score
                    file_num = audio_file.stem.replace('q', '')
                    if file_num.isdigit() and file_num in EXPECTED_TRANSCRIPTIONS:
                        expected = EXPECTED_TRANSCRIPTIONS[file_num]
                        similarity = self.calculate_similarity(expected, transcription.text)
                        result["expected_text"] = expected
                        result["quality_score"] = similarity
                        
                        if similarity >= 0.85:
                            quality_label = "EXCELLENT" if similarity >= 0.95 else "GOOD"
                            print(f"  ✅ Quality: {quality_label} ({similarity:.2%})")
                        else:
                            quality_label = "LOW"
                            print(f"  ⚠️  Quality: {quality_label} ({similarity:.2%})")
                    else:
                        # Q-files without expected text
                        result["quality_score"] = None
                        print(f"  ℹ️  TTS Output: {transcription.text}")
                    
                    # Cleanup
                    temp_audio.unlink()
                    
                except asyncio.TimeoutError:
                    result["error"] = f"Timeout after {responses_received} responses"
                    result["success"] = False
                    print(f"  ❌ Timeout waiting for voice!")
                    return result
                
                # Calculate total processing time
                total_time = upload_time + response_time + result.get("validation_time", 0)
                result["total_processing_time"] = total_time
                result["success"] = True
                
                print(f"  ⏱️  Total Time: {total_time:.2f}s")
                print(f"  ✅ Test PASSED")
                
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            print(f"  ❌ Test FAILED: {e}")
        
        return result
    
    async def run_batch_tests(self, file_limit: Optional[int] = None):
        """
        Run batch tests on all voice files.
        
        Args:
            file_limit: Optional limit on number of files to test
        """
        print("\n" + "="*70)
        print("🚀 BATCH WEBSOCKET VOICE TESTING")
        print("="*70)
        
        # Get all WebM files
        audio_files = sorted(self.voice_dir.glob("*.webm"))
        
        if file_limit:
            audio_files = audio_files[:file_limit]
        
        total_files = len(audio_files)
        print(f"\n📁 Found {total_files} audio files")
        print(f"🌐 WebSocket URL: {self.ws_url}")
        print(f"📍 Voice Directory: {self.voice_dir}")
        
        # Confirm start
        print(f"\n⏳ Starting batch tests in 3 seconds...")
        await asyncio.sleep(3)
        
        # Run tests
        start_time = time.time()
        
        for idx, audio_file in enumerate(audio_files, 1):
            result = await self.test_single_file(audio_file, idx, total_files)
            self.results.append(result)
            
            # Update summary
            if result["success"]:
                self.summary["passed"] += 1
            else:
                self.summary["failed"] += 1
            
            # Brief pause between tests
            if idx < total_files:
                await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        self.summary["total_tests"] = total_files
        self.summary["total_time"] = total_time
        
        # Calculate averages
        successful_results = [r for r in self.results if r["success"]]
        if successful_results:
            avg_processing = sum(r["total_processing_time"] for r in successful_results) / len(successful_results)
            self.summary["avg_processing_time"] = avg_processing
            
            quality_scores = [r["quality_score"] for r in successful_results if r.get("quality_score")]
            if quality_scores:
                self.summary["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate and display comprehensive test report."""
        print("\n" + "="*70)
        print("📊 BATCH TEST RESULTS")
        print("="*70)
        
        # Overall summary
        print(f"\n🎯 Overall Summary:")
        print(f"  Total Tests: {self.summary['total_tests']}")
        print(f"  ✅ Passed: {self.summary['passed']}")
        print(f"  ❌ Failed: {self.summary['failed']}")
        print(f"  Success Rate: {(self.summary['passed']/self.summary['total_tests']*100):.1f}%")
        print(f"  Total Time: {self.summary['total_time']:.2f}s")
        
        if self.summary['passed'] > 0:
            print(f"\n⏱️  Performance Metrics:")
            print(f"  Avg Processing Time: {self.summary['avg_processing_time']:.2f}s")
            
            if self.summary['avg_quality_score'] > 0:
                print(f"  Avg Quality Score: {self.summary['avg_quality_score']:.2%}")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        
        for result in self.results:
            status = "✅" if result["success"] else "❌"
            file_name = result["file"]
            
            if result["success"]:
                time_str = f"{result['total_processing_time']:.2f}s"
                quality_str = ""
                if result.get("quality_score"):
                    quality_str = f" | Quality: {result['quality_score']:.1%}"
                print(f"  {status} {file_name:15s} | Time: {time_str}{quality_str}")
            else:
                print(f"  {status} {file_name:15s} | Error: {result.get('error', 'Unknown')}")
        
        # Save detailed results
        output_file = Path(__file__).parent / f"batch_test_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": self.summary,
                "results": self.results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file.name}")
        print("\n" + "="*70)
        print("✅ Batch Testing Complete!")
        print("="*70 + "\n")


async def main():
    """Main execution function."""
    # Load environment
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file!")
        sys.exit(1)
    
    # Configuration
    ws_url = "ws://localhost:8000/api/v1/ws/simple-voice-chat"
    voice_dir = Path(__file__).parent / "voice_input_tests"
    
    # Optional: Limit number of tests (remove or set to None for all files)
    file_limit = None  # Change to a number like 5 for testing, or None for all 40 files
    
    # Run batch tests
    tester = BatchWebSocketTester(ws_url, voice_dir, api_key)
    await tester.run_batch_tests(file_limit=file_limit)


if __name__ == "__main__":
    asyncio.run(main())
