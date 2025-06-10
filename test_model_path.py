#!/usr/bin/env python3
"""
Simple test script to verify model path finding works
"""
import sys
import os
import json
from pathlib import Path

def test_model_path(model_name):
    """Test finding model path for a specific model."""
    print(f"🔍 Testing model path for: {model_name}")
    
    # Load model registry directly from JSON file
    registry_path = Path("beautyai_inference/config/model_registry.json")
    with open(registry_path, 'r') as f:
        registry_data = json.load(f)
    
    # Get model config from registry
    if model_name not in registry_data["models"]:
        print(f"❌ Model {model_name} not found in registry")
        return False
    
    model_data = registry_data["models"][model_name]
    model_id = model_data["model_id"]
    model_filename = model_data.get("model_filename", None)
    
    print(f"✅ Model ID: {model_id}")
    if model_filename:
        print(f"✅ Specific filename: {model_filename}")
    
    # Find GGUF file
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_id_safe = model_id.replace("/", "--")
    
    import glob
    search_patterns = [
        f"{cache_dir}/models--{model_id_safe}/snapshots/*/",
        f"{cache_dir}/models--{model_id_safe.replace('_', '--')}/snapshots/*/",
    ]
    
    print(f"🔍 Searching in: {search_patterns}")
    
    for pattern_dir in search_patterns:
        dirs = glob.glob(pattern_dir)
        print(f"Found directories: {len(dirs)}")
        
        for dir_path in dirs:
            print(f"Checking: {dir_path}")
            
            if model_filename:
                # Look for specific filename
                specific_path = os.path.join(dir_path, model_filename)
                print(f"Looking for specific file: {specific_path}")
                if os.path.exists(specific_path):
                    print(f"✅ Found model: {specific_path}")
                    return True
            
            # Look for any GGUF files
            gguf_files = glob.glob(os.path.join(dir_path, "*.gguf"))
            if gguf_files:
                print(f"Found GGUF files: {[os.path.basename(f) for f in gguf_files]}")
                # Prefer Q4_K_M or specific quantization
                for gguf_file in gguf_files:
                    if "Q4_K" in os.path.basename(gguf_file):
                        print(f"✅ Found preferred model: {gguf_file}")
                        return True
                
                # Use first file if no preferred found
                print(f"✅ Found model (first): {gguf_files[0]}")
                return True
    
    print(f"❌ Could not find GGUF file for {model_name}")
    return False

if __name__ == "__main__":
    # Test a few models
    models_to_test = [
        "qwen3-unsloth-q4ks",
        "bee1reason-arabic-q4ks"
    ]
    
    for model in models_to_test:
        print("\n" + "="*50)
        success = test_model_path(model)
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
