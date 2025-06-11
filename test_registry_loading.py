#!/usr/bin/env python3
"""Test script to check if the model registry loads successfully."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'beautyai_inference'))

from config.config_manager import ModelRegistry

def test_registry_loading():
    """Test loading the model registry to identify any field issues."""
    registry_path = "beautyai_inference/config/model_registry.json"
    
    try:
        registry = ModelRegistry.load_from_file(registry_path)
        print("✅ Model registry loaded successfully!")
        print(f"Default model: {registry.default_model}")
        print(f"Available models: {list(registry.models.keys())}")
        
        # Check specific models
        for model_name in ["qwen3-unsloth-q4ks", "qwen3-unsloth-q4km"]:
            if model_name in registry.models:
                model = registry.models[model_name]
                print(f"\n{model_name}:")
                print(f"  Engine: {model.engine_type}")
                print(f"  Model ID: {model.model_id}")
                print(f"  Quantization: {model.quantization}")
            else:
                print(f"❌ Model {model_name} not found in registry")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model registry: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    test_registry_loading()
