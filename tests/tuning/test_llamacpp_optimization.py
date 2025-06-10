#!/usr/bin/env python3
"""
Test script to benchmark the optimized LlamaCpp engine performance.
Run this script to test the speed improvements on RTX 4090.
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

def run_speed_test():
    """Run a speed test with the optimized LlamaCpp engine."""
    
    # Configuration for Bee1reason model (adjust as needed)
    model_config = ModelConfig(
        model_id="mradermacher/Bee1reason-arabic-Qwen-14B-i1-GGUF",
        engine_type="llama.cpp",
        model_filename="Bee1reason-arabic-Qwen-14B.i1-Q4_K_S.gguf",
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.8,
        name="bee1reason-test"
    )
    
    logger.info("🚀 Starting LlamaCpp Engine Speed Test (RTX 4090 Optimized)")
    logger.info(f"Model: {model_config.model_id}")
    logger.info(f"Target: 50-100+ tokens per second")
    print("-" * 60)
    
    try:
        # Initialize the engine
        engine = LlamaCppEngine(model_config)
        
        # Load the model
        logger.info("Loading model...")
        load_start = time.time()
        engine.load_model()
        load_time = time.time() - load_start
        logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Test prompts of varying lengths
        test_prompts = [
            "What is artificial intelligence?",
            "Explain the benefits of using quantized models in deep learning applications.",
            "Write a Python function that calculates the Fibonacci sequence up to n terms using dynamic programming approach.",
        ]
        
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n📝 Test {i}/3: {prompt[:50]}...")
            
            # Run benchmark
            start_time = time.time()
            result = engine.benchmark(prompt, max_new_tokens=100)
            end_time = time.time()
            
            # Extract metrics
            inference_time = result['inference_time']
            tokens_per_second = result['tokens_per_second']
            output_tokens = result['output_tokens']
            
            # Accumulate totals
            total_tokens += output_tokens
            total_time += inference_time
            
            # Display results
            logger.info(f"⚡ Inference time: {inference_time:.2f}s")
            logger.info(f"🎯 Tokens per second: {tokens_per_second:.1f}")
            logger.info(f"📊 Output tokens: {output_tokens}")
            
            # Performance assessment
            if tokens_per_second >= 100:
                status = "🏆 EXCELLENT"
            elif tokens_per_second >= 50:
                status = "✅ GOOD"
            elif tokens_per_second >= 25:
                status = "⚠️ ACCEPTABLE"
            else:
                status = "❌ NEEDS IMPROVEMENT"
            
            logger.info(f"Status: {status}")
        
        # Overall performance summary
        print("\n" + "=" * 60)
        logger.info("📊 OVERALL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        average_speed = total_tokens / total_time if total_time > 0 else 0
        logger.info(f"Average speed: {average_speed:.1f} tokens/second")
        logger.info(f"Total tokens generated: {total_tokens}")
        logger.info(f"Total inference time: {total_time:.2f}s")
        
        # Final assessment
        if average_speed >= 100:
            logger.info("🏆 OUTSTANDING: Achieved 100+ tokens/second target!")
        elif average_speed >= 50:
            logger.info("✅ SUCCESS: Achieved 50+ tokens/second target!")
        elif average_speed >= 25:
            logger.info("⚠️ MODERATE: Consider further optimization")
        else:
            logger.info("❌ UNDERPERFORMING: Significant optimization needed")
        
        # Memory stats
        memory_stats = engine.get_memory_stats()
        logger.info(f"GPU Memory Usage: {memory_stats}")
        
        # Unload model
        engine.unload_model()
        logger.info("🧹 Model unloaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_speed_test()
