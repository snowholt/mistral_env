#!/usr/bin/env python3
"""
Comprehensive Voice Performance Benchmarking System for BeautyAI Framework
===========================================================================

This script provides comprehensive benchmarking capabilities for voice WebSocket endpoints including:
1. Real voice input transcription before testing
2. Concurrent user load testing
3. Speed, latency, and accuracy benchmarking
4. Output voice transcription and accuracy analysis
5. Detailed performance reports with visualizations

Features:
- Uses real voice files from voice_tests/input_test_questions/
- Transcribes input voices using Whisper for baseline accuracy
- Tests concurrent WebSocket connections (1-100+ users)
- Measures end-to-end latency and server processing time
- Transcribes output audio and compares accuracy
- Generates detailed performance and accuracy reports
- Supports both Arabic and English benchmarking

Usage:
    python voice_performance_benchmark.py --help
    python voice_performance_benchmark.py --concurrent-users 50 --language ar
    python voice_performance_benchmark.py --full-accuracy-test --max-users 20

Author: BeautyAI Framework
Date: 2025-07-24
"""

import asyncio
import json
import base64
import time
import websockets
import argparse
import logging
import threading
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures

# Audio processing and transcription
import whisper
import librosa
import soundfile as sf

# Data analysis and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# System monitoring
import psutil
import torch

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AudioTestCase:
    """Represents a single audio test case."""
    name: str
    file_path: Path
    language: str
    expected_transcription: Optional[str] = None
    actual_input_transcription: Optional[str] = None
    file_size_bytes: int = 0
    duration_seconds: float = 0.0


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""
    test_name: str
    audio_file: str
    language: str
    voice_type: str
    success: bool
    start_time: float
    end_time: float
    total_duration: float
    server_processing_time: float
    connection_time: float
    transcription_received: Optional[str]
    response_text_received: Optional[str]
    output_audio_file: Optional[str]
    output_audio_size: int
    output_transcription: Optional[str]
    errors: List[str]
    memory_usage_mb: int
    accuracy_metrics: Dict[str, float]


@dataclass
class ConcurrentTestResult:
    """Results from concurrent user testing."""
    concurrent_users: int
    successful_connections: int
    failed_connections: int
    average_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_requests_per_second: float
    success_rate: float
    max_memory_usage_mb: int
    individual_results: List[BenchmarkResult]


class VoicePerformanceBenchmark:
    """
    Comprehensive performance benchmarking for voice services.
    
    Tests WebSocket voice endpoints across multiple metrics:
    - Response time and latency
    - Concurrent user handling
    - Transcription accuracy (input vs output)
    - Memory usage and system resource consumption
    - Audio quality analysis
    """
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        """Initialize the benchmark system."""
        self.base_url = base_url
        self.whisper_model = None
        self.results = []
        self.concurrent_results = []
        
        # Test configuration
        self.input_audio_dir = Path("voice_tests/input_test_questions")
        self.output_dir = Path("voice_tests/benchmark_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.memory_monitor = MemoryMonitor()
        
        logger.info(f"VoicePerformanceBenchmark initialized with URL: {base_url}")
    
    async def initialize(self):
        """Initialize the benchmark system including Whisper model."""
        logger.info("🔄 Initializing Whisper model for transcription accuracy testing...")
        try:
            # Load Whisper model for transcription accuracy testing
            self.whisper_model = whisper.load_model("turbo")  # Fast but accurate model
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def load_test_cases(self) -> List[AudioTestCase]:
        """Load and prepare audio test cases from input directory."""
        logger.info(f"📁 Loading test cases from {self.input_audio_dir}")
        
        test_cases = []
        audio_extensions = {'.wav', '.webm', '.mp3', '.flac', '.m4a'}
        
        for audio_file in self.input_audio_dir.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                try:
                    # Get file info
                    file_size = audio_file.stat().st_size
                    
                    # Load audio to get duration
                    audio_data, sample_rate = librosa.load(str(audio_file), sr=None)
                    duration = len(audio_data) / sample_rate
                    
                    # Determine language from filename
                    language = "ar" if "_ar" in audio_file.stem or "arabic" in audio_file.stem.lower() else "en"
                    
                    test_case = AudioTestCase(
                        name=audio_file.stem,
                        file_path=audio_file,
                        language=language,
                        file_size_bytes=file_size,
                        duration_seconds=duration
                    )
                    
                    test_cases.append(test_case)
                    logger.info(f"   ✅ Loaded: {audio_file.name} ({duration:.2f}s, {file_size} bytes, {language})")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to load {audio_file.name}: {e}")
        
        logger.info(f"📊 Loaded {len(test_cases)} test cases")
        return test_cases
    
    async def transcribe_input_audio(self, test_cases: List[AudioTestCase]) -> List[AudioTestCase]:
        """Transcribe input audio files for baseline accuracy comparison."""
        logger.info("🎤 Transcribing input audio files for accuracy baseline...")
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"   [{i}/{len(test_cases)}] Transcribing: {test_case.name}")
            
            try:
                # Transcribe using Whisper
                result = self.whisper_model.transcribe(
                    str(test_case.file_path),
                    language="ar" if test_case.language == "ar" else "en"
                )
                
                test_case.actual_input_transcription = result["text"].strip()
                logger.info(f"   ✅ Transcription: '{test_case.actual_input_transcription[:100]}...'")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to transcribe {test_case.name}: {e}")
                test_case.actual_input_transcription = f"[Transcription Error: {e}]"
        
        return test_cases
    
    async def single_websocket_test(
        self,
        test_case: AudioTestCase,
        voice_type: str = "female",
        user_id: int = 0
    ) -> BenchmarkResult:
        """
        Test a single WebSocket connection with an audio file.
        
        Args:
            test_case: Audio test case to use
            voice_type: Voice type (male/female)
            user_id: User identifier for concurrent testing
            
        Returns:
            BenchmarkResult with comprehensive metrics
        """
        test_name = f"{test_case.name}_{test_case.language}_{voice_type}_user{user_id}"
        
        # Initialize result
        start_time = time.time()
        result = BenchmarkResult(
            test_name=test_name,
            audio_file=str(test_case.file_path),
            language=test_case.language,
            voice_type=voice_type,
            success=False,
            start_time=start_time,
            end_time=0,
            total_duration=0,
            server_processing_time=0,
            connection_time=0,
            transcription_received=None,
            response_text_received=None,
            output_audio_file=None,
            output_audio_size=0,
            output_transcription=None,
            errors=[],
            memory_usage_mb=0,
            accuracy_metrics={}
        )
        
        try:
            # Monitor memory before test
            result.memory_usage_mb = self.memory_monitor.get_current_usage()["ram_process_mb"]
            
            # Read audio file
            with open(test_case.file_path, 'rb') as f:
                audio_data = f.read()
            
            # Build WebSocket URL
            ws_url = f"{self.base_url}/api/v1/ws/simple-voice-chat"
            ws_url += f"?language={test_case.language}&voice_type={voice_type}"
            
            # Connect to WebSocket
            connect_start = time.time()
            async with websockets.connect(ws_url) as websocket:
                result.connection_time = time.time() - connect_start
                
                # Wait for welcome message
                try:
                    welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    welcome_data = json.loads(welcome_msg)
                    
                    if welcome_data.get("type") != "connection_established":
                        result.errors.append(f"Unexpected welcome message: {welcome_data}")
                        return result
                        
                except asyncio.TimeoutError:
                    result.errors.append("Timeout waiting for welcome message")
                    return result
                
                # Send audio data
                send_time = time.time()
                await websocket.send(audio_data)
                
                # Wait for response
                async for message in websocket:
                    try:
                        message_data = json.loads(message)
                        message_type = message_data.get("type")
                        
                        if message_type == "voice_response":
                            response_time = time.time()
                            result.total_duration = response_time - start_time
                            result.server_processing_time = message_data.get("response_time_ms", 0) / 1000
                            result.transcription_received = message_data.get("transcription", "")
                            result.response_text_received = message_data.get("response_text", "")
                            
                            # Handle audio response
                            audio_base64 = message_data.get("audio_base64")
                            if audio_base64:
                                await self._process_output_audio(audio_base64, result, test_case)
                            
                            # Calculate accuracy metrics
                            result.accuracy_metrics = self._calculate_accuracy_metrics(
                                test_case.actual_input_transcription,
                                result.transcription_received,
                                result.output_transcription
                            )
                            
                            result.success = True
                            break
                            
                        elif message_type == "error":
                            result.errors.append(f"Server error: {message_data.get('message', 'Unknown')}")
                            break
                    
                    except json.JSONDecodeError:
                        continue  # Skip non-JSON messages
                    
                    # Timeout check
                    if time.time() - send_time > 30:
                        result.errors.append("Response timeout")
                        break
        
        except Exception as e:
            result.errors.append(f"Connection error: {e}")
        
        finally:
            result.end_time = time.time()
            if result.total_duration == 0:
                result.total_duration = result.end_time - result.start_time
        
        return result
    
    async def _process_output_audio(
        self,
        audio_base64: str,
        result: BenchmarkResult,
        test_case: AudioTestCase
    ):
        """Process and transcribe output audio for accuracy analysis."""
        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_base64)
            result.output_audio_size = len(audio_bytes)
            
            # Save output audio
            timestamp = int(time.time())
            output_file = self.output_dir / f"output_{result.test_name}_{timestamp}.wav"
            
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)
            
            result.output_audio_file = str(output_file)
            
            # Transcribe output audio for accuracy comparison
            if self.whisper_model:
                transcription_result = self.whisper_model.transcribe(
                    str(output_file),
                    language="ar" if test_case.language == "ar" else "en"
                )
                result.output_transcription = transcription_result["text"].strip()
            
        except Exception as e:
            result.errors.append(f"Output audio processing error: {e}")
    
    def _calculate_accuracy_metrics(
        self,
        input_transcription: Optional[str],
        received_transcription: Optional[str],
        output_transcription: Optional[str]
    ) -> Dict[str, float]:
        """Calculate various accuracy metrics."""
        metrics = {
            "input_transcription_similarity": 0.0,
            "output_transcription_quality": 0.0,
            "end_to_end_accuracy": 0.0
        }
        
        try:
            if input_transcription and received_transcription:
                # Simple similarity calculation (could be enhanced with more sophisticated methods)
                metrics["input_transcription_similarity"] = self._calculate_text_similarity(
                    input_transcription, received_transcription
                )
            
            if received_transcription and output_transcription:
                metrics["output_transcription_quality"] = self._calculate_text_similarity(
                    received_transcription, output_transcription
                )
            
            if input_transcription and output_transcription:
                metrics["end_to_end_accuracy"] = self._calculate_text_similarity(
                    input_transcription, output_transcription
                )
        
        except Exception as e:
            logger.warning(f"Error calculating accuracy metrics: {e}")
        
        return metrics
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (can be enhanced with BLEU, etc.)."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def concurrent_user_test(
        self,
        test_cases: List[AudioTestCase],
        max_concurrent_users: int = 50,
        test_duration_seconds: int = 60,
        language: str = "ar",
        voice_type: str = "female"
    ) -> ConcurrentTestResult:
        """
        Test concurrent user handling capabilities.
        
        Args:
            test_cases: Audio test cases to use
            max_concurrent_users: Maximum number of concurrent users to test
            test_duration_seconds: Duration of concurrent test
            language: Language to test
            voice_type: Voice type to test
            
        Returns:
            ConcurrentTestResult with comprehensive metrics
        """
        logger.info(f"🚀 Starting concurrent user test: {max_concurrent_users} users, {test_duration_seconds}s")
        
        # Filter test cases by language
        filtered_cases = [tc for tc in test_cases if tc.language == language]
        if not filtered_cases:
            raise ValueError(f"No test cases found for language: {language}")
        
        # Monitor memory during test
        self.memory_monitor.start_monitoring()
        
        start_time = time.time()
        tasks = []
        results = []
        
        try:
            # Create concurrent tasks
            for user_id in range(max_concurrent_users):
                # Cycle through test cases
                test_case = filtered_cases[user_id % len(filtered_cases)]
                
                # Create concurrent task
                task = asyncio.create_task(
                    self.single_websocket_test(test_case, voice_type, user_id)
                )
                tasks.append(task)
                
                # Small delay to stagger connections
                await asyncio.sleep(0.1)
            
            # Wait for all tasks with timeout
            timeout = test_duration_seconds + 30  # Extra buffer
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results
            successful_results = []
            failed_count = 0
            
            for task_result in completed_tasks:
                if isinstance(task_result, BenchmarkResult):
                    results.append(task_result)
                    if task_result.success:
                        successful_results.append(task_result)
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                    logger.error(f"Task failed with exception: {task_result}")
        
        except asyncio.TimeoutError:
            logger.error(f"Concurrent test timed out after {timeout} seconds")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        finally:
            self.memory_monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = time.time() - start_time
        successful_count = len(successful_results)
        
        if successful_results:
            response_times = [r.total_duration for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else avg_response_time
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else avg_response_time
            throughput = successful_count / total_time
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0.0
            throughput = 0.0
        
        concurrent_result = ConcurrentTestResult(
            concurrent_users=max_concurrent_users,
            successful_connections=successful_count,
            failed_connections=failed_count,
            average_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput_requests_per_second=throughput,
            success_rate=successful_count / max_concurrent_users if max_concurrent_users > 0 else 0.0,
            max_memory_usage_mb=self.memory_monitor.get_peak_usage(),
            individual_results=results
        )
        
        self.concurrent_results.append(concurrent_result)
        
        logger.info(f"✅ Concurrent test completed:")
        logger.info(f"   Users: {max_concurrent_users}")
        logger.info(f"   Successful: {successful_count}")
        logger.info(f"   Failed: {failed_count}")
        logger.info(f"   Success Rate: {concurrent_result.success_rate:.2%}")
        logger.info(f"   Avg Response Time: {avg_response_time:.2f}s")
        logger.info(f"   Throughput: {throughput:.2f} req/s")
        
        return concurrent_result
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("📊 Generating comprehensive performance report...")
        
        report = {
            "summary": {
                "test_date": datetime.now().isoformat(),
                "total_individual_tests": len(self.results),
                "total_concurrent_tests": len(self.concurrent_results),
                "benchmark_version": "1.0.0"
            },
            "individual_tests": {
                "successful_tests": len([r for r in self.results if r.success]),
                "failed_tests": len([r for r in self.results if not r.success]),
                "average_response_time": 0.0,
                "accuracy_metrics": {}
            },
            "concurrent_tests": [],
            "system_performance": {
                "peak_memory_usage_mb": 0,
                "average_memory_usage_mb": 0
            },
            "recommendations": []
        }
        
        # Process individual test results
        if self.results:
            successful_results = [r for r in self.results if r.success]
            if successful_results:
                response_times = [r.total_duration for r in successful_results]
                report["individual_tests"]["average_response_time"] = statistics.mean(response_times)
                
                # Aggregate accuracy metrics
                accuracy_keys = successful_results[0].accuracy_metrics.keys()
                for key in accuracy_keys:
                    values = [r.accuracy_metrics.get(key, 0.0) for r in successful_results]
                    report["individual_tests"]["accuracy_metrics"][key] = statistics.mean(values)
        
        # Process concurrent test results
        for concurrent_result in self.concurrent_results:
            report["concurrent_tests"].append({
                "concurrent_users": concurrent_result.concurrent_users,
                "success_rate": concurrent_result.success_rate,
                "average_response_time": concurrent_result.average_response_time,
                "throughput_rps": concurrent_result.throughput_requests_per_second,
                "p95_response_time": concurrent_result.p95_response_time,
                "max_memory_usage_mb": concurrent_result.max_memory_usage_mb
            })
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        # Save report to file
        report_file = self.output_dir / f"performance_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Performance report saved to: {report_file}")
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []
        
        avg_response_time = report["individual_tests"]["average_response_time"]
        if avg_response_time > 3.0:
            recommendations.append("⚠️ Average response time exceeds 3 seconds - consider optimizing TTS/ASR pipeline")
        elif avg_response_time < 2.0:
            recommendations.append("✅ Excellent response time performance - meets target <2s requirement")
        
        # Analyze concurrent performance
        for concurrent_test in report["concurrent_tests"]:
            if concurrent_test["success_rate"] < 0.8:
                recommendations.append(f"⚠️ Low success rate ({concurrent_test['success_rate']:.1%}) at {concurrent_test['concurrent_users']} users - check server capacity")
            elif concurrent_test["success_rate"] > 0.95:
                recommendations.append(f"✅ Excellent success rate at {concurrent_test['concurrent_users']} concurrent users")
        
        # Memory usage recommendations
        max_memory = max([ct["max_memory_usage_mb"] for ct in report["concurrent_tests"]], default=0)
        if max_memory > 1000:
            recommendations.append("⚠️ High memory usage detected - consider memory optimization")
        
        return recommendations
    
    def generate_visualization_charts(self):
        """Generate performance visualization charts."""
        logger.info("📈 Generating performance visualization charts...")
        
        if not self.concurrent_results:
            logger.warning("No concurrent test results to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BeautyAI Voice Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # Chart 1: Response Time vs Concurrent Users
        users = [r.concurrent_users for r in self.concurrent_results]
        avg_times = [r.average_response_time for r in self.concurrent_results]
        p95_times = [r.p95_response_time for r in self.concurrent_results]
        
        axes[0, 0].plot(users, avg_times, 'o-', label='Average Response Time', linewidth=2)
        axes[0, 0].plot(users, p95_times, 's--', label='P95 Response Time', linewidth=2, alpha=0.7)
        axes[0, 0].axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Target (2s)')
        axes[0, 0].set_xlabel('Concurrent Users')
        axes[0, 0].set_ylabel('Response Time (seconds)')
        axes[0, 0].set_title('Response Time vs Load')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Chart 2: Success Rate vs Concurrent Users
        success_rates = [r.success_rate * 100 for r in self.concurrent_results]
        axes[0, 1].plot(users, success_rates, 'o-', color='green', linewidth=2)
        axes[0, 1].axhline(y=95, color='r', linestyle='--', alpha=0.5, label='Target (95%)')
        axes[0, 1].set_xlabel('Concurrent Users')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Success Rate vs Load')
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Chart 3: Throughput vs Concurrent Users
        throughput = [r.throughput_requests_per_second for r in self.concurrent_results]
        axes[1, 0].plot(users, throughput, 'o-', color='purple', linewidth=2)
        axes[1, 0].set_xlabel('Concurrent Users')
        axes[1, 0].set_ylabel('Throughput (req/s)')
        axes[1, 0].set_title('Throughput vs Load')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Chart 4: Memory Usage vs Concurrent Users
        memory_usage = [r.max_memory_usage_mb for r in self.concurrent_results]
        axes[1, 1].plot(users, memory_usage, 'o-', color='orange', linewidth=2)
        axes[1, 1].set_xlabel('Concurrent Users')
        axes[1, 1].set_ylabel('Peak Memory Usage (MB)')
        axes[1, 1].set_title('Memory Usage vs Load')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save charts
        chart_file = self.output_dir / f"performance_charts_{int(time.time())}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Performance charts saved to: {chart_file}")
        
        plt.show()


class MemoryMonitor:
    """Monitor system memory usage during benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.peak_memory = 0
        self.memory_samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self.monitoring:
            usage = self.get_current_usage()
            self.memory_samples.append(usage)
            self.peak_memory = max(self.peak_memory, usage["ram_process_mb"])
            time.sleep(0.5)  # Sample every 500ms
    
    def get_current_usage(self) -> Dict[str, int]:
        """Get current memory usage."""
        process = psutil.Process()
        return {
            "ram_total_mb": psutil.virtual_memory().total // 1024 // 1024,
            "ram_used_mb": psutil.virtual_memory().used // 1024 // 1024,
            "ram_process_mb": process.memory_info().rss // 1024 // 1024,
            "gpu_memory_mb": self._get_gpu_memory() if torch.cuda.is_available() else 0
        }
    
    def _get_gpu_memory(self) -> int:
        """Get GPU memory usage if available."""
        try:
            return torch.cuda.memory_allocated() // 1024 // 1024
        except:
            return 0
    
    def get_peak_usage(self) -> int:
        """Get peak memory usage during monitoring."""
        return self.peak_memory


async def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Comprehensive Voice Performance Benchmarking")
    parser.add_argument("--url", default="ws://localhost:8000", help="WebSocket base URL")
    parser.add_argument("--language", default="ar", choices=["ar", "en"], help="Language to test")
    parser.add_argument("--voice", default="female", choices=["male", "female"], help="Voice type")
    parser.add_argument("--concurrent-users", type=int, default=10, help="Max concurrent users to test")
    parser.add_argument("--test-duration", type=int, default=60, help="Concurrent test duration (seconds)")
    parser.add_argument("--full-accuracy-test", action="store_true", help="Run full accuracy analysis")
    parser.add_argument("--generate-charts", action="store_true", help="Generate visualization charts")
    parser.add_argument("--output-dir", help="Custom output directory")
    
    args = parser.parse_args()
    
    # Initialize benchmark system
    benchmark = VoicePerformanceBenchmark(base_url=args.url)
    
    if args.output_dir:
        benchmark.output_dir = Path(args.output_dir)
        benchmark.output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize Whisper model
        await benchmark.initialize()
        
        # Load test cases
        test_cases = benchmark.load_test_cases()
        if not test_cases:
            logger.error("No test cases found!")
            return
        
        # Transcribe input audio for accuracy baseline
        if args.full_accuracy_test:
            test_cases = await benchmark.transcribe_input_audio(test_cases)
        
        # Run concurrent user tests
        logger.info(f"🚀 Starting comprehensive benchmark test...")
        
        # Test different concurrent user levels
        user_levels = [1, 5, 10, 20, args.concurrent_users] if args.concurrent_users > 20 else [1, 5, args.concurrent_users]
        user_levels = sorted(set(user_levels))  # Remove duplicates and sort
        
        for user_count in user_levels:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {user_count} concurrent users")
            logger.info(f"{'='*60}")
            
            try:
                await benchmark.concurrent_user_test(
                    test_cases=test_cases,
                    max_concurrent_users=user_count,
                    test_duration_seconds=args.test_duration,
                    language=args.language,
                    voice_type=args.voice
                )
            except Exception as e:
                logger.error(f"Failed concurrent test with {user_count} users: {e}")
        
        # Generate comprehensive report
        report = benchmark.generate_performance_report()
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 BENCHMARK SUMMARY")
        logger.info(f"{'='*80}")
        
        for concurrent_test in report["concurrent_tests"]:
            logger.info(f"👥 {concurrent_test['concurrent_users']} Users:")
            logger.info(f"   ✅ Success Rate: {concurrent_test['success_rate']:.1%}")
            logger.info(f"   ⏱️  Avg Response: {concurrent_test['average_response_time']:.2f}s")
            logger.info(f"   🚀 Throughput: {concurrent_test['throughput_rps']:.2f} req/s")
            logger.info(f"   💾 Peak Memory: {concurrent_test['max_memory_usage_mb']} MB")
        
        # Display recommendations
        if report["recommendations"]:
            logger.info(f"\n📋 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                logger.info(f"   {rec}")
        
        # Generate visualization charts
        if args.generate_charts:
            benchmark.generate_visualization_charts()
        
        logger.info(f"\n✅ Benchmark completed successfully!")
        logger.info(f"📁 Results saved to: {benchmark.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
