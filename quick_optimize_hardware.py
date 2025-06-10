#!/usr/bin/env python3
"""
Quick Hardware Optimization Script for BeautyAI
Finds optimal llama.cpp settings using binary search instead of brute force.
Optimizes for RTX 4090 and similar high-end GPUs.
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

from beautyai_inference.config.config_manager import AppConfig
from beautyai_inference.services.model.registry_service import RegistryService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareOptimizer:
    """Quick hardware optimization using binary search and smart heuristics."""
    
    def __init__(self, model_name: str = "mradermacher/Bee1reason-arabic-Qwen-14B-i1-GGUF"):
        self.model_name = model_name
        self.model_path = None
        self.optimization_results = []
        self.best_settings = None
        
        # Hardware detection
        self.detect_hardware()
        
        # Find model path
        self.find_model_path()
    
    def detect_hardware(self):
        """Detect hardware capabilities."""
        logger.info("🔍 Detecting hardware capabilities...")
        
        # CPU detection
        self.cpu_cores = os.cpu_count()
        self.cpu_threads = psutil.cpu_count(logical=True)
        
        # Memory detection
        self.system_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_available = True
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_compute_capability = torch.cuda.get_device_capability(0)
        else:
            self.gpu_available = False
            self.gpu_name = "None"
            self.gpu_memory_gb = 0
            self.gpu_compute_capability = (0, 0)
        
        logger.info(f"📊 Hardware Summary:")
        logger.info(f"   CPU: {self.cpu_cores} cores, {self.cpu_threads} threads")
        logger.info(f"   RAM: {self.system_ram_gb:.1f} GB")
        logger.info(f"   GPU: {self.gpu_name}")
        logger.info(f"   VRAM: {self.gpu_memory_gb:.1f} GB")
        logger.info(f"   Compute: {self.gpu_compute_capability}")
    
    def find_model_path(self):
        """Find the GGUF model file."""
        logger.info(f"🔍 Finding model path for: {self.model_name}")
        
        try:
            # Load model registry directly from JSON file
            registry_path = Path(__file__).parent / "beautyai_inference" / "config" / "model_registry.json"
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Get model config from registry
            if self.model_name not in registry_data["models"]:
                raise ValueError(f"Model {self.model_name} not found in registry")
            
            model_data = registry_data["models"][self.model_name]
            model_id = model_data["model_id"]
            model_filename = model_data.get("model_filename", None)
            
            logger.info(f"🔍 Looking for model_id: {model_id}")
            if model_filename:
                logger.info(f"🔍 Specific filename: {model_filename}")
            
            # Find GGUF file
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_id_safe = model_id.replace("/", "--")
            
            import glob
            search_patterns = [
                f"{cache_dir}/models--{model_id_safe}/snapshots/*/",
                f"{cache_dir}/models--{model_id_safe.replace('_', '--')}/snapshots/*/",
            ]
            
            for pattern_dir in search_patterns:
                dirs = glob.glob(pattern_dir)
                for dir_path in dirs:
                    if model_filename:
                        # Look for specific filename
                        specific_path = os.path.join(dir_path, model_filename)
                        if os.path.exists(specific_path):
                            self.model_path = specific_path
                            logger.info(f"✅ Found model: {self.model_path}")
                            return
                    
                    # Look for any GGUF files
                    gguf_files = glob.glob(os.path.join(dir_path, "*.gguf"))
                    if gguf_files:
                        # Prefer Q4_K_M or specific quantization
                        for gguf_file in gguf_files:
                            if "Q4_K" in os.path.basename(gguf_file):
                                self.model_path = gguf_file
                                logger.info(f"✅ Found model: {self.model_path}")
                                return
                        
                        # Use first file if no preferred found
                        self.model_path = gguf_files[0]
                        logger.info(f"✅ Found model: {self.model_path}")
                        return
            
            raise FileNotFoundError(f"Could not find GGUF file for {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error finding model path: {e}")
            raise
    
    def get_hardware_based_ranges(self) -> Dict[str, List[int]]:
        """Get optimization ranges based on detected hardware."""
        if self.gpu_memory_gb >= 20:  # RTX 4090 class
            return {
                "n_batch": [512, 1024, 2048, 4096, 8192, 16384],
                "n_threads": [8, 12, 16, 20, 24, 28, 32],
                "n_ctx": [512, 1024, 2048, 4096],
                "n_gpu_layers": [-1]  # All layers on GPU
            }
        elif self.gpu_memory_gb >= 12:  # RTX 4070 Ti class
            return {
                "n_batch": [256, 512, 1024, 2048, 4096],
                "n_threads": [8, 12, 16, 20, 24],
                "n_ctx": [512, 1024, 2048],
                "n_gpu_layers": [-1, 40, 30]
            }
        elif self.gpu_memory_gb >= 8:  # RTX 4060 Ti class
            return {
                "n_batch": [128, 256, 512, 1024, 2048],
                "n_threads": [4, 8, 12, 16],
                "n_ctx": [512, 1024],
                "n_gpu_layers": [30, 20, 10]
            }
        else:  # CPU or low-end GPU
            return {
                "n_batch": [64, 128, 256, 512],
                "n_threads": [4, 8, 12, min(16, self.cpu_threads)],
                "n_ctx": [512, 1024],
                "n_gpu_layers": [0] if not self.gpu_available else [10, 5]
            }
    
    def test_configuration(self, settings: Dict[str, int], test_duration: int = 30) -> Optional[Dict[str, float]]:
        """Test a specific configuration and return performance metrics."""
        logger.info(f"🧪 Testing: {settings}")
        
        try:
            # Initialize model with test settings
            model = Llama(
                model_path=self.model_path,
                n_gpu_layers=settings["n_gpu_layers"],
                n_ctx=settings["n_ctx"],
                n_batch=settings["n_batch"],
                n_threads=settings["n_threads"],
                n_threads_batch=settings["n_threads"],
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
            
            # Test prompt
            test_prompt = "Explain quantum computing in simple terms."
            
            # Warmup
            _ = model(test_prompt, max_tokens=10)
            
            # Actual test
            start_time = time.time()
            response = model(
                test_prompt,
                max_tokens=100,
                temperature=0.1,
                top_p=0.8,
                echo=False,
            )
            end_time = time.time()
            
            # Calculate metrics
            if response and 'choices' in response:
                output_text = response['choices'][0]['text']
                output_tokens = len(output_text.split())
                inference_time = end_time - start_time
                tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
                
                # Memory usage
                gpu_memory_used = 0
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                
                result = {
                    "tokens_per_second": tokens_per_second,
                    "inference_time": inference_time,
                    "output_tokens": output_tokens,
                    "gpu_memory_gb": gpu_memory_used,
                    "stable": True
                }
                
                # Cleanup
                del model
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"✅ Result: {tokens_per_second:.1f} tok/s, {inference_time:.2f}s")
                return result
            
        except Exception as e:
            logger.warning(f"❌ Configuration failed: {e}")
            return {"tokens_per_second": 0, "stable": False, "error": str(e)}
        
        return None
    
    def binary_search_optimal_batch(self, fixed_settings: Dict[str, int]) -> int:
        """Find optimal batch size using binary search."""
        logger.info("🔍 Binary search for optimal batch size...")
        
        ranges = self.get_hardware_based_ranges()
        batch_candidates = ranges["n_batch"]
        
        low, high = 0, len(batch_candidates) - 1
        best_speed = 0
        best_batch = batch_candidates[0]
        
        while low <= high:
            mid = (low + high) // 2
            test_batch = batch_candidates[mid]
            
            test_settings = fixed_settings.copy()
            test_settings["n_batch"] = test_batch
            
            result = self.test_configuration(test_settings)
            
            if result and result.get("stable", False):
                speed = result["tokens_per_second"]
                if speed > best_speed:
                    best_speed = speed
                    best_batch = test_batch
                    low = mid + 1  # Try larger batch
                else:
                    high = mid - 1  # Try smaller batch
            else:
                high = mid - 1  # Configuration failed, try smaller
        
        logger.info(f"🎯 Optimal batch size: {best_batch} ({best_speed:.1f} tok/s)")
        return best_batch
    
    def optimize_threads(self, fixed_settings: Dict[str, int]) -> int:
        """Find optimal thread count."""
        logger.info("🔍 Optimizing thread count...")
        
        ranges = self.get_hardware_based_ranges()
        thread_candidates = ranges["n_threads"]
        
        best_speed = 0
        best_threads = thread_candidates[0]
        
        for threads in thread_candidates:
            test_settings = fixed_settings.copy()
            test_settings["n_threads"] = threads
            
            result = self.test_configuration(test_settings)
            
            if result and result.get("stable", False):
                speed = result["tokens_per_second"]
                if speed > best_speed:
                    best_speed = speed
                    best_threads = threads
                    
                logger.info(f"   {threads} threads: {speed:.1f} tok/s")
        
        logger.info(f"🎯 Optimal threads: {best_threads} ({best_speed:.1f} tok/s)")
        return best_threads
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run complete optimization process."""
        logger.info("🚀 Starting hardware optimization...")
        
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available")
        
        ranges = self.get_hardware_based_ranges()
        
        # Start with conservative baseline
        baseline_settings = {
            "n_batch": ranges["n_batch"][1] if len(ranges["n_batch"]) > 1 else ranges["n_batch"][0],
            "n_threads": ranges["n_threads"][1] if len(ranges["n_threads"]) > 1 else ranges["n_threads"][0],
            "n_ctx": ranges["n_ctx"][0],
            "n_gpu_layers": ranges["n_gpu_layers"][0]
        }
        
        logger.info(f"📊 Baseline settings: {baseline_settings}")
        baseline_result = self.test_configuration(baseline_settings)
        
        if not baseline_result or not baseline_result.get("stable", False):
            raise RuntimeError("Baseline configuration failed")
        
        # Step 1: Optimize batch size
        optimal_batch = self.binary_search_optimal_batch(baseline_settings)
        
        # Step 2: Optimize threads with optimal batch
        optimized_settings = baseline_settings.copy()
        optimized_settings["n_batch"] = optimal_batch
        optimal_threads = self.optimize_threads(optimized_settings)
        
        # Final optimal settings
        final_settings = {
            "n_batch": optimal_batch,
            "n_threads": optimal_threads,
            "n_ctx": baseline_settings["n_ctx"],
            "n_gpu_layers": baseline_settings["n_gpu_layers"]
        }
        
        # Test final configuration
        logger.info("🎯 Testing final optimized configuration...")
        final_result = self.test_configuration(final_settings)
        
        optimization_summary = {
            "hardware": {
                "cpu_cores": self.cpu_cores,
                "cpu_threads": self.cpu_threads,
                "system_ram_gb": self.system_ram_gb,
                "gpu_name": self.gpu_name,
                "gpu_memory_gb": self.gpu_memory_gb
            },
            "baseline": {
                "settings": baseline_settings,
                "performance": baseline_result
            },
            "optimized": {
                "settings": final_settings,
                "performance": final_result
            },
            "improvement": {
                "speed_multiplier": final_result["tokens_per_second"] / baseline_result["tokens_per_second"]
                if baseline_result["tokens_per_second"] > 0 else 0,
                "time_reduction_percent": (1 - final_result["inference_time"] / baseline_result["inference_time"]) * 100
                if baseline_result["inference_time"] > 0 else 0
            }
        }
        
        self.best_settings = final_settings
        return optimization_summary
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save optimization results to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hardware_optimization_{timestamp}.json"
        
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filepath}")
    
    def update_llamacpp_engine(self):
        """Update the llamacpp_engine.py with optimized settings."""
        if not self.best_settings:
            logger.error("No optimization results available")
            return
        
        engine_file = Path(__file__).parent / "beautyai_inference" / "inference_engines" / "llamacpp_engine.py"
        
        if not engine_file.exists():
            logger.error(f"Engine file not found: {engine_file}")
            return
        
        # Read current file
        with open(engine_file, 'r') as f:
            content = f.read()
        
        # Update the optimized parameters
        updates = [
            (f"n_ctx = 1024", f"n_ctx = {self.best_settings['n_ctx']}"),
            (f"n_batch = 8192", f"n_batch = {self.best_settings['n_batch']}"),
            (f"n_threads = 24", f"n_threads = {self.best_settings['n_threads']}"),
            (f"n_threads_batch = 24", f"n_threads_batch = {self.best_settings['n_threads']}"),
        ]
        
        for old, new in updates:
            if old in content:
                content = content.replace(old, new)
                logger.info(f"✅ Updated: {old} → {new}")
        
        # Backup original file
        backup_file = engine_file.with_suffix('.py.backup')
        if not backup_file.exists():
            import shutil
            shutil.copy2(engine_file, backup_file)
            logger.info(f"📋 Backup created: {backup_file}")
        
        # Write updated file
        with open(engine_file, 'w') as f:
            f.write(content)
        
        logger.info(f"🔄 Updated engine file: {engine_file}")


def main():
    """Main optimization routine."""
    print("🚀 BeautyAI Hardware Optimizer")
    print("=" * 50)
    
    # Load available GGUF models from registry
    try:
        config_path = Path(__file__).parent / "beautyai_inference" / "config" / "model_registry.json"
        with open(config_path, 'r') as f:
            registry = json.load(f)
        
        # Filter GGUF models only
        gguf_models = []
        for name, config in registry["models"].items():
            if config.get("engine_type") == "llama.cpp":
                gguf_models.append((name, config.get("description", name)))
        
        if not gguf_models:
            print("❌ No GGUF models found in registry")
            return 1
            
        print("Available GGUF models:")
        for i, (name, desc) in enumerate(gguf_models, 1):
            print(f"  {i}. {name}")
            print(f"     {desc}")
    
    except Exception as e:
        print(f"❌ Error loading model registry: {e}")
        return 1
    
    try:
        choice = input(f"\nSelect model (1-{len(gguf_models)}) [1]: ").strip()
        model_idx = int(choice) - 1 if choice else 0
        selected_model = gguf_models[model_idx][0]  # Get model name, not description
    except (ValueError, IndexError):
        selected_model = gguf_models[0][0]  # Default to first model name
    
    print(f"\n🎯 Optimizing for: {selected_model}")
    
    # Run optimization
    optimizer = HardwareOptimizer(selected_model)
    
    try:
        results = optimizer.run_optimization()
        
        print("\n" + "=" * 50)
        print("📊 OPTIMIZATION RESULTS")
        print("=" * 50)
        
        baseline = results["baseline"]["performance"]
        optimized = results["optimized"]["performance"]
        
        print(f"Baseline Speed:    {baseline['tokens_per_second']:.1f} tokens/sec")
        print(f"Optimized Speed:   {optimized['tokens_per_second']:.1f} tokens/sec")
        print(f"Speed Improvement: {results['improvement']['speed_multiplier']:.2f}x")
        print(f"Time Reduction:    {results['improvement']['time_reduction_percent']:.1f}%")
        
        print(f"\n🎯 Optimal Settings:")
        for key, value in results["optimized"]["settings"].items():
            print(f"  {key}: {value}")
        
        # Save results
        optimizer.save_results(results)
        
        # Ask to update engine
        update = input("\n🔄 Update llamacpp_engine.py with optimal settings? (y/N): ").strip().lower()
        if update in ['y', 'yes']:
            optimizer.update_llamacpp_engine()
            print("✅ Engine updated successfully!")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
