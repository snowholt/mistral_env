#!/usr/bin/env python3
"""
Comprehensive model testing script for all models in the registry.
Tests speed and accuracy comparison across all available models with standardized parameters.
"""

import sys
import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from beautyai_inference.config.config_manager import ModelConfig, ModelRegistry
from beautyai_inference.inference_engines.llamacpp_engine import LlamaCppEngine
from beautyai_inference.inference_engines.transformers_engine import TransformersEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standardized test parameters for fair comparison
STANDARD_PARAMS = {
    "temperature": 0.1,
    "max_new_tokens": 256,
    "top_p": 0.8,
    "top_k": 10,
    "repetition_penalty": 1.05,
    "do_sample": True
}

# Test prompts for accuracy and speed evaluation - sorted by language
TEST_PROMPTS = [
    # English prompts
    {
        "id": "simple_qa",
        "prompt": "What is the difference between Botox and other alternatives?",
        "category": "cosmetic_comparison",
        "expected_length": "long"
    },
    {
        "id": "comparison",
        "prompt": "What should be avoided after mesotherapy?",
        "category": "post_treatment_care",
        "expected_length": "medium"
    },
    # Arabic prompts
    {
        "id": "reasoning",
        "prompt": "ما الفرق بين البوتوكس وبدائل أخرى؟",
        "category": "cosmetic_comparison",
        "expected_length": "long"
    },
    {
        "id": "technical",
        "prompt": "هل يمكن تكرار التخلص من السيلوليت بشكل منتظم؟",
        "category": "treatment_frequency",
        "expected_length": "medium"
    },
    {
        "id": "practical",
        "prompt": "من هو أفضل مرشح لـ الميزوثيرابي؟",
        "category": "candidate_suitability",
        "expected_length": "medium"
    }
]

def load_model_registry():
    """Load the model registry from the config file."""
    registry_path = Path(__file__).parent.parent.parent / "beautyai_inference" / "config" / "model_registry.json"
    logger.info(f"Loading model registry from: {registry_path}")
    
    if not registry_path.exists():
        logger.error(f"Model registry file not found: {registry_path}")
        return None
        
    return ModelRegistry.load_from_file(registry_path)

def create_engine(model_config):
    """Create the appropriate engine based on model configuration."""
    if model_config.engine_type == "llama.cpp":
        return LlamaCppEngine(model_config)
    elif model_config.engine_type == "transformers":
        return TransformersEngine(model_config)
    else:
        raise ValueError(f"Unsupported engine type: {model_config.engine_type}")

def standardize_model_config(model_config):
    """Create a standardized version of the model config for fair testing."""
    standardized = ModelConfig(
        model_id=model_config.model_id,
        engine_type=model_config.engine_type,
        quantization=model_config.quantization,
        dtype=model_config.dtype,
        model_filename=model_config.model_filename,
        name=model_config.name,
        description=model_config.description,
        model_architecture=model_config.model_architecture,
        max_new_tokens=STANDARD_PARAMS['max_new_tokens'],
        temperature=STANDARD_PARAMS['temperature'],
        top_p=STANDARD_PARAMS['top_p'],
        do_sample=STANDARD_PARAMS['do_sample'],
        custom_generation_params={
            "top_k": STANDARD_PARAMS['top_k'],
            "repetition_penalty": STANDARD_PARAMS['repetition_penalty']
        }
    )
    return standardized

def test_single_model(model_config):
    """Test a single model with all test prompts."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🧪 TESTING MODEL: {model_config.name}")
    logger.info(f"📋 ID: {model_config.model_id}")
    logger.info(f"🔧 Engine: {model_config.engine_type}")
    logger.info(f"⚙️ Quantization: {model_config.quantization}")
    logger.info(f"{'='*80}")
    
    # Standardize the config for fair testing
    std_config = standardize_model_config(model_config)
    
    model_results = {
        "model_name": model_config.name,
        "model_id": model_config.model_id,
        "engine_type": model_config.engine_type,
        "quantization": model_config.quantization,
        "dtype": model_config.dtype,
        "model_filename": model_config.model_filename,
        "test_parameters": STANDARD_PARAMS,
        "load_time": 0,
        "unload_time": 0,
        "total_test_time": 0,
        "prompts_tested": [],
        "performance_metrics": {},
        "error": None,
        "status": "pending"
    }
    
    try:
        # Create and load the engine
        logger.info("🔄 Creating engine...")
        engine = create_engine(std_config)
        
        logger.info("📥 Loading model...")
        load_start = time.time()
        engine.load_model()
        load_time = time.time() - load_start
        model_results["load_time"] = load_time
        logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Test all prompts
        test_start = time.time()
        prompt_results = []
        total_tokens = 0
        total_inference_time = 0
        
        for i, test_prompt in enumerate(TEST_PROMPTS, 1):
            logger.info(f"\n🔍 Test {i}/5: {test_prompt['category']} - {test_prompt['id']}")
            logger.info(f"📝 Prompt: {test_prompt['prompt']}")
            
            try:
                # Test with standardized parameters
                start_time = time.time()
                
                # Use the appropriate method based on engine type
                if std_config.engine_type == "llama.cpp":
                    # For llama.cpp engines, use chat method with messages
                    messages = [{"role": "user", "content": test_prompt['prompt']}]
                    response = engine.chat(messages, max_new_tokens=STANDARD_PARAMS['max_new_tokens'])
                else:
                    # For transformers engines, use generate method
                    generation_params = {
                        "max_new_tokens": STANDARD_PARAMS['max_new_tokens'],
                        "temperature": STANDARD_PARAMS['temperature'],
                        "top_p": STANDARD_PARAMS['top_p'],
                        "do_sample": STANDARD_PARAMS['do_sample']
                    }
                    # Add custom params if available
                    if std_config.custom_generation_params:
                        generation_params.update(std_config.custom_generation_params)
                    
                    response = engine.generate(test_prompt['prompt'], **generation_params)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Calculate metrics
                response_tokens = len(response.split()) if response else 0
                tokens_per_second = response_tokens / inference_time if inference_time > 0 else 0
                
                prompt_result = {
                    "prompt_id": test_prompt['id'],
                    "prompt_text": test_prompt['prompt'],
                    "category": test_prompt['category'],
                    "response": response[:200] + "..." if len(response) > 200 else response,
                    "response_length": len(response),
                    "response_tokens": response_tokens,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                    "status": "success"
                }
                
                total_tokens += response_tokens
                total_inference_time += inference_time
                
                logger.info(f"✅ Response length: {len(response)} chars, {response_tokens} tokens")
                logger.info(f"⚡ Speed: {tokens_per_second:.1f} tokens/second")
                logger.info(f"⏱️ Time: {inference_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"❌ Prompt test failed: {e}")
                prompt_result = {
                    "prompt_id": test_prompt['id'],
                    "prompt_text": test_prompt['prompt'],
                    "category": test_prompt['category'],
                    "error": str(e),
                    "status": "failed"
                }
            
            prompt_results.append(prompt_result)
        
        test_end = time.time()
        total_test_time = test_end - test_start
        
        # Calculate overall performance metrics
        successful_tests = [p for p in prompt_results if p.get("status") == "success"]
        if successful_tests:
            avg_tokens_per_second = sum(p["tokens_per_second"] for p in successful_tests) / len(successful_tests)
            max_tokens_per_second = max(p["tokens_per_second"] for p in successful_tests)
            min_tokens_per_second = min(p["tokens_per_second"] for p in successful_tests)
            total_response_length = sum(p["response_length"] for p in successful_tests)
            avg_response_length = total_response_length / len(successful_tests)
        else:
            avg_tokens_per_second = 0
            max_tokens_per_second = 0
            min_tokens_per_second = 0
            avg_response_length = 0
        
        model_results.update({
            "total_test_time": total_test_time,
            "prompts_tested": prompt_results,
            "performance_metrics": {
                "successful_tests": len(successful_tests),
                "failed_tests": len(prompt_results) - len(successful_tests),
                "total_tokens_generated": total_tokens,
                "total_inference_time": total_inference_time,
                "avg_tokens_per_second": avg_tokens_per_second,
                "max_tokens_per_second": max_tokens_per_second,
                "min_tokens_per_second": min_tokens_per_second,
                "avg_response_length": avg_response_length,
                "speed_consistency": (min_tokens_per_second / max_tokens_per_second * 100) if max_tokens_per_second > 0 else 0
            },
            "status": "completed"
        })
        
        logger.info(f"\n📊 Model Performance Summary:")
        logger.info(f"✅ Successful tests: {len(successful_tests)}/5")
        logger.info(f"⚡ Average speed: {avg_tokens_per_second:.1f} tokens/second")
        logger.info(f"📏 Average response length: {avg_response_length:.0f} characters")
        
        # Unload the model
        logger.info("\n🧹 Unloading model...")
        unload_start = time.time()
        engine.unload_model()
        unload_time = time.time() - unload_start
        model_results["unload_time"] = unload_time
        logger.info(f"✅ Model unloaded in {unload_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        model_results.update({
            "error": str(e),
            "status": "failed"
        })
        import traceback
        traceback.print_exc()
    
    return model_results

def run_comprehensive_model_test():
    """Run comprehensive testing of all models in the registry."""
    logger.info("🚀 COMPREHENSIVE MODEL COMPARISON TEST")
    logger.info("Testing all models with standardized parameters for fair comparison")
    logger.info(f"Parameters: temp={STANDARD_PARAMS['temperature']}, max_tokens={STANDARD_PARAMS['max_new_tokens']}")
    print("=" * 80)
    
    # Load model registry
    registry = load_model_registry()
    if not registry:
        logger.error("❌ Failed to load model registry")
        return
    
    models = registry.list_models()
    logger.info(f"📋 Found {len(models)} models to test: {', '.join(models)}")
    
    # Prepare results structure
    test_results = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_parameters": STANDARD_PARAMS,
            "models_tested": len(models),
            "test_prompts": [p["id"] for p in TEST_PROMPTS]
        },
        "model_results": {},
        "summary": {}
    }
    
    successful_models = []
    failed_models = []
    
    # Test each model
    for i, model_name in enumerate(models, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"🔄 TESTING MODEL {i}/{len(models)}: {model_name}")
        logger.info(f"{'#'*80}")
        
        model_config = registry.get_model(model_name)
        if not model_config:
            logger.error(f"❌ Could not find model config for: {model_name}")
            continue
        
        model_result = test_single_model(model_config)
        test_results["model_results"][model_name] = model_result
        
        if model_result["status"] == "completed":
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # Add a delay between models to ensure clean unloading
        time.sleep(2)
    
    # Generate summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("📊 GENERATING SUMMARY STATISTICS")
    logger.info(f"{'='*80}")
    
    if successful_models:
        successful_results = [test_results["model_results"][name] for name in successful_models]
        
        # Speed comparison
        speeds = [r["performance_metrics"]["avg_tokens_per_second"] for r in successful_results]
        speed_rankings = sorted(
            [(name, test_results["model_results"][name]["performance_metrics"]["avg_tokens_per_second"]) 
             for name in successful_models], 
            key=lambda x: x[1], reverse=True
        )
        
        # Response quality (length as proxy)
        response_lengths = [r["performance_metrics"]["avg_response_length"] for r in successful_results]
        
        # Load times
        load_times = [r["load_time"] for r in successful_results]
        
        summary = {
            "successful_models": len(successful_models),
            "failed_models": len(failed_models),
            "speed_statistics": {
                "fastest_model": speed_rankings[0] if speed_rankings else None,
                "slowest_model": speed_rankings[-1] if speed_rankings else None,
                "avg_speed_all_models": sum(speeds) / len(speeds) if speeds else 0,
                "max_speed": max(speeds) if speeds else 0,
                "min_speed": min(speeds) if speeds else 0,
                "speed_rankings": speed_rankings
            },
            "response_statistics": {
                "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
                "max_response_length": max(response_lengths) if response_lengths else 0,
                "min_response_length": min(response_lengths) if response_lengths else 0
            },
            "load_time_statistics": {
                "avg_load_time": sum(load_times) / len(load_times) if load_times else 0,
                "fastest_load": min(load_times) if load_times else 0,
                "slowest_load": max(load_times) if load_times else 0
            }
        }
        
        test_results["summary"] = summary
        
        # Display summary
        logger.info(f"✅ Successful models: {len(successful_models)}")
        logger.info(f"❌ Failed models: {len(failed_models)}")
        
        if speed_rankings:
            logger.info(f"\n🏆 SPEED RANKINGS:")
            for i, (model, speed) in enumerate(speed_rankings, 1):
                logger.info(f"  {i}. {model}: {speed:.1f} tokens/second")
        
        logger.info(f"\n📊 OVERALL STATISTICS:")
        logger.info(f"  Average speed: {summary['speed_statistics']['avg_speed_all_models']:.1f} tokens/second")
        logger.info(f"  Average load time: {summary['load_time_statistics']['avg_load_time']:.2f} seconds")
        logger.info(f"  Average response length: {summary['response_statistics']['avg_response_length']:.0f} characters")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent / f"model_comparison_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
    
    logger.info(f"\n🎉 COMPREHENSIVE MODEL TEST COMPLETED!")
    logger.info(f"📈 Tested {len(models)} models with {len(TEST_PROMPTS)} prompts each")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_model_test()
