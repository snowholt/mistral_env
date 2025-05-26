"""
Benchmark service for performance testing of models.

This service handles comprehensive performance benchmarking including:
- Latency testing across different input sizes
- Throughput measurement 
- Memory usage monitoring
- Multi-run statistical analysis
- Result export and reporting
"""
import logging
import time
import json
from typing import Dict, Any, List

from ..base.base_service import BaseService
from ...config.config_manager import ModelConfig
from ...core.model_manager import ModelManager
from ...utils.memory_utils import get_gpu_memory_stats

logger = logging.getLogger(__name__)


class BenchmarkService(BaseService):
    """Service for performance benchmarking operations."""
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
    
    def run_benchmark(self, model_name: str, model_config: ModelConfig,
                     benchmark_config: Dict[str, Any]) -> int:
        """
        Run a comprehensive benchmark on the model.
        
        Args:
            model_name: Name of the model to benchmark
            model_config: Model configuration
            benchmark_config: Benchmark parameters
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            # Extract benchmark parameters
            num_runs = benchmark_config.get("num_runs", 3)
            input_lengths = benchmark_config.get("input_lengths", [10, 100, 1000])
            output_length = benchmark_config.get("output_length", 100)
            output_file = benchmark_config.get("output_file")
            
            print(f"\n=== Benchmarking {model_name} ===")
            print(f"Model ID: {model_config.model_id}")
            print(f"Engine: {model_config.engine_type}")
            print(f"Quantization: {model_config.quantization or 'none'}")
            print(f"Number of runs: {num_runs}")
            print(f"Input lengths: {input_lengths}")
            print(f"Output length: {output_length}")
            
            # Ensure the model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
                
            # Configure generation parameters (optimized for benchmark)
            generation_config = {
                "max_new_tokens": output_length,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
            }
            
            # Run the benchmark for each input length
            all_results = {}
            
            for input_length in input_lengths:
                print(f"\n--- Testing input length: {input_length} tokens ---")
                
                # Generate benchmark prompt
                prompt = "Hello, " * (input_length // 2)  # Simple repeating pattern to reach desired length
                
                # Run the benchmark multiple times
                results = []
                
                for i in range(num_runs):
                    print(f"Run {i+1}/{num_runs}...")
                    
                    # Clear GPU memory stats before test
                    start_memory = get_gpu_memory_stats()
                    
                    # Run generation
                    start_time = time.time()
                    _ = model.generate(prompt, generation_config)
                    end_time = time.time()
                    
                    # Calculate results
                    generation_time = end_time - start_time
                    output_length_actual = len(_.split()) if _ else 0
                    
                    # Get memory stats after generation
                    end_memory = get_gpu_memory_stats()
                    
                    memory_used_before = start_memory[0]["memory_used_mb"] if start_memory else 0
                    memory_used_after = end_memory[0]["memory_used_mb"] if end_memory else 0
                    memory_delta = memory_used_after - memory_used_before
                    
                    # Append results
                    results.append({
                        "run_id": i+1,
                        "time": generation_time,
                        "tokens_per_sec": output_length_actual / generation_time if generation_time > 0 else 0,
                        "memory_before": memory_used_before,
                        "memory_after": memory_used_after,
                        "memory_delta": memory_delta,
                        "output_tokens": output_length_actual
                    })
                    
                    print(f"Time: {generation_time:.2f}s, Tokens/s: {output_length_actual/generation_time:.2f}")
                
                # Calculate average metrics for this input length
                avg_time = sum(result["time"] for result in results) / len(results)
                avg_tokens_per_sec = sum(result["tokens_per_sec"] for result in results) / len(results)
                avg_memory_delta = sum(result["memory_delta"] for result in results) / len(results)
                
                print(f"\nAverage for {input_length} tokens input:")
                print(f"  Generation time: {avg_time:.2f}s")
                print(f"  Tokens per second: {avg_tokens_per_sec:.2f}")
                print(f"  Memory increase: {avg_memory_delta:.2f} MB")
                
                all_results[input_length] = {
                    "runs": results,
                    "summary": {
                        "avg_time": avg_time,
                        "avg_tokens_per_sec": avg_tokens_per_sec,
                        "avg_memory_delta": avg_memory_delta,
                        "std_time": self._calculate_std([r["time"] for r in results]),
                        "std_tokens_per_sec": self._calculate_std([r["tokens_per_sec"] for r in results])
                    }
                }
            
            # Save results to file if requested
            if output_file:
                try:
                    results_obj = {
                        "model": model_name,
                        "model_id": model_config.model_id,
                        "engine": model_config.engine_type,
                        "quantization": model_config.quantization,
                        "output_length": output_length,
                        "timestamp": int(time.time()),
                        "results": all_results,
                        "overall_summary": self._calculate_overall_summary(all_results)
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(results_obj, f, indent=2)
                        
                    print(f"\nResults saved to {output_file}")
                    
                except Exception as e:
                    return self._handle_error(e, f"Failed to save benchmark results to {output_file}")
            
            print("\n=== Benchmark Complete ===")
            return 0
            
        except Exception as e:
            return self._handle_error(e, f"Failed to benchmark model {model_name}")
    
    def run_latency_benchmark(self, model_name: str, model_config: ModelConfig,
                             prompts: List[str], num_runs: int = 5) -> Dict[str, Any]:
        """
        Run a focused latency benchmark with specific prompts.
        
        Args:
            model_name: Name of the model to benchmark
            model_config: Model configuration
            prompts: List of prompts to test
            num_runs: Number of runs per prompt
            
        Returns:
            Dict containing latency benchmark results
        """
        try:
            # Ensure the model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return {"success": False, "error": "Failed to load model"}
            
            results = {
                "model_name": model_name,
                "benchmark_type": "latency",
                "num_runs": num_runs,
                "prompt_results": [],
                "summary": {}
            }
            
            generation_config = {
                "max_new_tokens": 50,  # Fixed short output for latency testing
                "temperature": 0.0,    # Deterministic for consistency
                "do_sample": False
            }
            
            all_latencies = []
            
            for i, prompt in enumerate(prompts):
                print(f"\nTesting prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
                
                prompt_latencies = []
                
                for run in range(num_runs):
                    start_time = time.time()
                    model.generate(prompt, generation_config)
                    end_time = time.time()
                    
                    latency = end_time - start_time
                    prompt_latencies.append(latency)
                    all_latencies.append(latency)
                
                prompt_result = {
                    "prompt": prompt,
                    "latencies": prompt_latencies,
                    "avg_latency": sum(prompt_latencies) / len(prompt_latencies),
                    "min_latency": min(prompt_latencies),
                    "max_latency": max(prompt_latencies),
                    "std_latency": self._calculate_std(prompt_latencies)
                }
                
                results["prompt_results"].append(prompt_result)
                print(f"  Avg latency: {prompt_result['avg_latency']:.3f}s")
            
            # Calculate overall summary
            results["summary"] = {
                "overall_avg_latency": sum(all_latencies) / len(all_latencies),
                "overall_min_latency": min(all_latencies),
                "overall_max_latency": max(all_latencies),
                "overall_std_latency": self._calculate_std(all_latencies),
                "total_prompts": len(prompts),
                "total_runs": len(all_latencies)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running latency benchmark: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def run_throughput_benchmark(self, model_name: str, model_config: ModelConfig,
                                duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Run a throughput benchmark measuring tokens per second over time.
        
        Args:
            model_name: Name of the model to benchmark
            model_config: Model configuration
            duration_seconds: How long to run the benchmark
            
        Returns:
            Dict containing throughput benchmark results
        """
        try:
            # Ensure the model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return {"success": False, "error": "Failed to load model"}
            
            print(f"\nRunning throughput benchmark for {duration_seconds} seconds...")
            
            generation_config = {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True
            }
            
            test_prompt = "Write a short story about artificial intelligence and its impact on society."
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            total_tokens = 0
            total_requests = 0
            request_times = []
            
            while time.time() < end_time:
                request_start = time.time()
                response = model.generate(test_prompt, generation_config)
                request_end = time.time()
                
                tokens_generated = len(response.split())
                total_tokens += tokens_generated
                total_requests += 1
                request_times.append(request_end - request_start)
                
                if total_requests % 5 == 0:
                    elapsed = time.time() - start_time
                    current_throughput = total_tokens / elapsed
                    print(f"  {total_requests} requests, {total_tokens} tokens, {current_throughput:.2f} tokens/sec")
            
            actual_duration = time.time() - start_time
            
            results = {
                "model_name": model_name,
                "benchmark_type": "throughput",
                "duration_seconds": actual_duration,
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "avg_tokens_per_second": total_tokens / actual_duration,
                "avg_requests_per_second": total_requests / actual_duration,
                "avg_tokens_per_request": total_tokens / total_requests,
                "avg_request_time": sum(request_times) / len(request_times),
                "request_times": request_times
            }
            
            print(f"\nThroughput Results:")
            print(f"  Total requests: {total_requests}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Throughput: {results['avg_tokens_per_second']:.2f} tokens/sec")
            print(f"  Request rate: {results['avg_requests_per_second']:.2f} requests/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running throughput benchmark: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _ensure_model_loaded(self, model_name: str, model_config: ModelConfig):
        """
        Ensure the model is loaded and return it.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            
        Returns:
            The loaded model
        """
        if not self.model_manager.is_model_loaded(model_name):
            print(f"Loading model '{model_name}'...")
            self.model_manager.load_model(model_name, model_config)
            print(f"Model loaded successfully.")
        else:
            print(f"Using already loaded model '{model_name}'.")
            
        return self.model_manager.get_model(model_name)
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _calculate_overall_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall benchmark summary statistics."""
        all_times = []
        all_tokens_per_sec = []
        
        for input_length, result_data in all_results.items():
            for run in result_data["runs"]:
                all_times.append(run["time"])
                all_tokens_per_sec.append(run["tokens_per_sec"])
        
        if not all_times:
            return {}
        
        return {
            "overall_avg_time": sum(all_times) / len(all_times),
            "overall_avg_tokens_per_sec": sum(all_tokens_per_sec) / len(all_tokens_per_sec),
            "overall_min_time": min(all_times),
            "overall_max_time": max(all_times),
            "overall_std_time": self._calculate_std(all_times),
            "overall_std_tokens_per_sec": self._calculate_std(all_tokens_per_sec),
            "total_runs": len(all_times)
        }
