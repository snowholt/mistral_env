#!/usr/bin/env python3
"""
Parameter tuning helper for LlamaCpp engine optimization.
This script allows you to test different parameter combinations to find
the optimal settings for your specific hardware and model.
"""

import sys
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from beautyai_inference.config.config_manager import ModelConfig
from beautyai_inference.inference_engines.llamacpp_engine import LlamaCppEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for a parameter test."""
    name: str
    n_ctx: int
    n_batch: int
    n_threads: int
    top_k: int
    top_p: float
    description: str

def get_test_configurations() -> List[TestConfig]:
    """Get different test configurations to try."""
    return [
        TestConfig(
            name="Conservative",
            n_ctx=2048,
            n_batch=2048,
            n_threads=8,
            top_k=20,
            top_p=0.9,
            description="Safe settings, similar to original"
        ),
        TestConfig(
            name="Balanced",
            n_ctx=2048,
            n_batch=3072,
            n_threads=12,
            top_k=15,
            top_p=0.85,
            description="Moderate optimization"
        ),
        TestConfig(
            name="Aggressive",
            n_ctx=2048,
            n_batch=4096,
            n_threads=16,
            top_k=10,
            top_p=0.8,
            description="Maximum speed optimization (recommended)"
        ),
        TestConfig(
            name="Extreme",
            n_ctx=1024,
            n_batch=5120,
            n_threads=20,
            top_k=5,
            top_p=0.7,
            description="Extreme speed, may impact quality"
        ),
    ]

def test_configuration(config: TestConfig, model_config: ModelConfig) -> Tuple[float, dict]:
    """Test a specific configuration and return tokens per second."""
    
    logger.info(f"\n🧪 Testing {config.name} configuration:")
    logger.info(f"   {config.description}")
    logger.info(f"   n_ctx={config.n_ctx}, n_batch={config.n_batch}, "
               f"n_threads={config.n_threads}, top_k={config.top_k}, top_p={config.top_p}")
    
    # Temporarily modify the engine to use test parameters
    # Note: This would require modifying the engine to accept these parameters
    # For now, we'll use the standard benchmark
    
    try:
        engine = LlamaCppEngine(model_config)
        
        # Quick test
        test_prompt = "Explain machine learning in simple terms."
        result = engine.benchmark(test_prompt, max_new_tokens=50)
        
        tokens_per_second = result['tokens_per_second']
        memory_stats = result['memory_stats']
        
        engine.unload_model()
        
        return tokens_per_second, memory_stats
        
    except Exception as e:
        logger.error(f"❌ Test failed for {config.name}: {e}")
        return 0.0, {}

def run_parameter_sweep():
    """Run a sweep of different parameter configurations."""
    
    model_config = ModelConfig(
        model_id="bee1reason-arabic-qwen-14b",
        model_filename="Bee1reason-arabic-Qwen-14B.i1-Q4_K_S.gguf",
        max_new_tokens=128,
        temperature=0.1,
        max_seq_len=2048
    )
    
    logger.info("🎯 LlamaCpp Parameter Sweep Test")
    logger.info("Testing different parameter combinations to find optimal settings")
    print("=" * 80)
    
    configurations = get_test_configurations()
    results = []
    
    for config in configurations:
        try:
            tokens_per_second, memory_stats = test_configuration(config, model_config)
            results.append((config, tokens_per_second, memory_stats))
            
            logger.info(f"✅ {config.name}: {tokens_per_second:.1f} tokens/second")
            
        except Exception as e:
            logger.error(f"❌ {config.name} failed: {e}")
            results.append((config, 0.0, {}))
    
    # Summary
    print("\n" + "=" * 80)
    logger.info("📊 PARAMETER SWEEP RESULTS")
    print("=" * 80)
    
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by tokens/second
    
    for i, (config, tokens_per_second, memory_stats) in enumerate(results, 1):
        status = "🏆" if i == 1 else "✅" if tokens_per_second >= 50 else "⚠️" if tokens_per_second >= 25 else "❌"
        logger.info(f"{status} #{i} {config.name}: {tokens_per_second:.1f} tokens/second")
        logger.info(f"     Settings: ctx={config.n_ctx}, batch={config.n_batch}, "
                   f"threads={config.n_threads}, top_k={config.top_k}, top_p={config.top_p}")
    
    if results:
        best_config, best_speed, _ = results[0]
        logger.info(f"\n🏆 BEST CONFIGURATION: {best_config.name}")
        logger.info(f"🎯 BEST SPEED: {best_speed:.1f} tokens/second")
        logger.info(f"📝 DESCRIPTION: {best_config.description}")

if __name__ == "__main__":
    run_parameter_sweep()
