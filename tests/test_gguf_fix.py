#!/usr/bin/env python3
"""Test script to verify GGUF model loading after the fix."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'beautyai_inference'))

from config.config_manager import ModelRegistry
from core.model_factory import create_model_engine

def test_gguf_model_loading():
    """Test loading GGUF models to ensure they use llama.cpp engine."""
    print("Testing GGUF model loading after fix...")
    
    # Load registry
    registry_path = "beautyai_inference/config/model_registry.json"
    registry = ModelRegistry.load_from_file(registry_path)
    
    # Test GGUF models
    gguf_models = ["qwen3-unsloth-q4ks", "qwen3-unsloth-q4km"]
    
    for model_name in gguf_models:
        print(f"\n--- Testing {model_name} ---")
        
        if model_name not in registry.models:
            print(f"❌ Model {model_name} not found in registry")
            continue
            
        model_config = registry.models[model_name]
        print(f"Registry engine type: {model_config.engine_type}")
        print(f"Model ID: {model_config.model_id}")
        print(f"Model filename: {model_config.model_filename}")
        
        try:
            # This should now create a LlamaCppEngine instead of TransformersEngine
            engine = create_model_engine(model_config)
            engine_type = type(engine).__name__
            print(f"✅ Created engine: {engine_type}")
            
            if "LlamaCpp" in engine_type:
                print("✅ Correctly using LlamaCpp engine for GGUF model")
            else:
                print(f"❌ Unexpected engine type: {engine_type}")
                
        except Exception as e:
            print(f"❌ Failed to create engine: {e}")
            print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_gguf_model_loading()
