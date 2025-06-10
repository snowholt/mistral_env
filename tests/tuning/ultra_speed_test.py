#!/usr/bin/env python3
"""
Ultra-speed test for the maximally optimized LlamaCpp engine.
This script tests the absolute maximum speed achievable with aggressive optimizations.
"""

import sys
import os
import time
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from beautyai_inference.config.config_manager import ModelConfig
from beautyai_inference.inference_engines.llamacpp_engine import LlamaCppEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ultra_speed_test():
    """Run ultra-speed test with maximally aggressive optimizations."""
    
    # Configuration for ultra-speed
    model_config = ModelConfig(
        model_id="mradermacher/Bee1reason-arabic-Qwen-14B-i1-GGUF",
        engine_type="llama.cpp",
        quantization="i1-Q4_K_S",
        dtype="float16",
        max_new_tokens=64,
        model_filename="Bee1reason-arabic-Qwen-14B.i1-Q4_K_S.gguf",
        custom_generation_params={
            "temperature": 0.01,
            "top_p": 0.5,
            "top_k": 1,
            "repetition_penalty": 1.0
        }
    )
    
    logger.info("🚀 ULTRA-SPEED LlamaCpp Engine Test (RTX 4090 Max Optimized)")
    logger.info(f"Model: {model_config.model_id}")
    logger.info(f"Target: 80-120+ tokens per second")
    print("-" * 60)
    
    try:
        # Initialize the engine
        engine = LlamaCppEngine(model_config)
        
        # Load the model
        logger.info("Loading model with ultra-aggressive settings...")
        load_start = time.time()
        engine.load_model()
        load_time = time.time() - load_start
        logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Short test prompts for maximum speed
        test_prompts = [
            "What is AI?",
            "Define machine learning.",
            "Explain neural networks briefly.",
            "What is deep learning?",
            "Define artificial intelligence.",
        ]
        
        total_tokens = 0
        total_time = 0
        max_speed = 0
        speeds = []
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n⚡ Speed Test {i}/5: {prompt}")
            
            # Run ultra-fast benchmark
            start_time = time.time()
            result = engine.benchmark(prompt, max_new_tokens=32)  # Very short for speed
            end_time = time.time()
            
            # Extract metrics
            inference_time = result['inference_time']
            tokens_per_second = result['tokens_per_second']
            output_tokens = result['output_tokens']
            
            # Track records
            if tokens_per_second > max_speed:
                max_speed = tokens_per_second
            speeds.append(tokens_per_second)
            
            # Accumulate totals
            total_tokens += output_tokens
            total_time += inference_time
            
            # Display results
            logger.info(f"🎯 Speed: {tokens_per_second:.1f} tokens/second")
            logger.info(f"📊 Tokens: {output_tokens} in {inference_time:.2f}s")
        
        # Performance analysis
        print("\n" + "=" * 60)
        logger.info("🏆 ULTRA-SPEED PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        average_speed = total_tokens / total_time if total_time > 0 else 0
        min_speed = min(speeds)
        
        logger.info(f"Average speed: {average_speed:.1f} tokens/second")
        logger.info(f"Maximum speed: {max_speed:.1f} tokens/second")
        logger.info(f"Minimum speed: {min_speed:.1f} tokens/second")
        logger.info(f"Speed consistency: {((min_speed/max_speed)*100):.1f}%")
        
        # Ultra-performance assessment
        if average_speed >= 120:
            logger.info("🏆 LEGENDARY: 120+ tokens/second achieved!")
        elif average_speed >= 100:
            logger.info("🔥 EXCEPTIONAL: 100+ tokens/second achieved!")
        elif average_speed >= 80:
            logger.info("⚡ EXCELLENT: 80+ tokens/second achieved!")
        elif average_speed >= 60:
            logger.info("✅ VERY GOOD: 60+ tokens/second achieved!")
        else:
            logger.info("⚠️ Need further optimization")
        
        # Test chat method for comparison
        logger.info("\n🔄 Testing chat method speed...")
        chat_start = time.time()
        chat_result = engine.chat([{"role": "user", "content": "What is AI?"}], max_new_tokens=32)
        chat_time = time.time() - chat_start
        chat_tokens = len(chat_result.split())
        chat_speed = chat_tokens / chat_time if chat_time > 0 else 0
        
        logger.info(f"Chat method speed: {chat_speed:.1f} tokens/second")
        
        # Speed parity check
        speed_ratio = chat_speed / average_speed if average_speed > 0 else 0
        if speed_ratio >= 0.95:
            logger.info("✅ Perfect speed parity between benchmark and chat!")
        elif speed_ratio >= 0.8:
            logger.info("✅ Good speed parity between benchmark and chat")
        else:
            logger.info(f"⚠️ Speed difference: chat is {speed_ratio*100:.1f}% of benchmark speed")
        
        # Memory stats
        memory_stats = engine.get_memory_stats()
        if memory_stats:
            gpu_util = memory_stats[0].get('gpu_utilization', 0)
            logger.info(f"GPU utilization: {gpu_util}%")
        
        # Unload model
        engine.unload_model()
        logger.info("🧹 Model unloaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Ultra-speed test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_ultra_speed_test()
