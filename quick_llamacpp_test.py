#!/usr/bin/env python3
"""
Quick test script using models from your registry.
"""

import sys
import os
import time
import logging
import json

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from beautyai_inference.config.config_manager import ModelConfig
from beautyai_inference.inference_engines.llamacpp_engine import LlamaCppEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_registry_models():
    """Load available llamacpp models from registry."""
    registry_path = "beautyai_inference/config/model_registry.json"
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        llamacpp_models = {}
        for name, config in registry["models"].items():
            if config.get("engine_type") == "llama.cpp":
                llamacpp_models[name] = config
        
        return llamacpp_models
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        return {}

def quick_test(model_name: str, model_config_dict: dict):
    """Quick performance test of a model."""
    
    model_config = ModelConfig(
        model_id=model_config_dict["model_id"],
        engine_type=model_config_dict["engine_type"],
        model_filename=model_config_dict.get("model_filename"),
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.8,
        name=model_name
    )
    
    logger.info(f"🧪 Testing {model_name}")
    logger.info(f"   Model ID: {model_config.model_id}")
    logger.info(f"   Quantization: {model_config_dict.get('quantization', 'N/A')}")
    
    try:
        engine = LlamaCppEngine(model_config)
        
        # Load model
        start_time = time.time()
        engine.load_model()
        load_time = time.time() - start_time
        
        # Quick benchmark
        result = engine.benchmark("Explain artificial intelligence briefly.", max_new_tokens=50)
        
        # Results
        tokens_per_second = result['tokens_per_second']
        memory_stats = result['memory_stats'][0] if result['memory_stats'] else {}
        
        logger.info(f"✅ Load time: {load_time:.2f}s")
        logger.info(f"⚡ Speed: {tokens_per_second:.1f} tokens/second")
        logger.info(f"💾 GPU Memory: {memory_stats.get('memory_used_mb', 0):.0f}MB")
        
        engine.unload_model()
        return tokens_per_second
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 0.0

def main():
    """Test all available llamacpp models."""
    
    print("🚀 Quick LlamaCpp Model Performance Test")
    print("=" * 60)
    
    models = load_registry_models()
    if not models:
        print("No llamacpp models found in registry.")
        return
    
    results = []
    
    for name, config in models.items():
        try:
            speed = quick_test(name, config)
            results.append((name, speed, config.get('quantization', 'N/A')))
            print("-" * 40)
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
            break
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            results.append((name, 0.0, config.get('quantization', 'N/A')))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, speed, quant) in enumerate(results, 1):
        status = "🏆" if speed >= 100 else "✅" if speed >= 50 else "⚠️" if speed >= 25 else "❌"
        print(f"{status} #{i} {name[:35]:<35} {speed:>6.1f} tok/s ({quant})")
    
    if results:
        best_name, best_speed, best_quant = results[0]
        print(f"\n🏆 Best Model: {best_name} ({best_speed:.1f} tok/s)")

if __name__ == "__main__":
    main()
