#!/usr/bin/env python3
"""
Test script to verify GGUF model loading and registry fixes.
"""
import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, '/home/lumi/beautyai')

from beautyai_inference.config.config_manager import AppConfig, ModelConfig
from beautyai_inference.core.model_factory import ModelFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_registry_loading():
    """Test that the model registry loads without errors."""
    print("=" * 60)
    print("Testing Model Registry Loading")
    print("=" * 60)
    
    try:
        # Load the config which should load the model registry
        config = AppConfig()
        config.models_file = "/home/lumi/beautyai/beautyai_inference/config/model_registry.json"
        config.load_model_registry()
        
        print(f"✅ Model registry loaded successfully!")
        print(f"📋 Default model: {config.model_registry.default_model}")
        print(f"📊 Total models: {len(config.model_registry.models)}")
        
        # List all models
        for name, model_config in config.model_registry.models.items():
            print(f"   • {name}: {model_config.model_id} ({model_config.engine_type})")
            if hasattr(model_config, 'tokenizer_model_id') and model_config.tokenizer_model_id:
                print(f"     └─ Tokenizer: {model_config.tokenizer_model_id}")
        
        return config
        
    except Exception as e:
        print(f"❌ Failed to load model registry: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_gguf_model_creation():
    """Test creating a GGUF model through the model factory."""
    print("\n" + "=" * 60)
    print("Testing GGUF Model Creation")
    print("=" * 60)
    
    config = test_model_registry_loading()
    if not config:
        return False
    
    # Try to get the qwen3-unsloth-q4ks model
    qwen_model = config.model_registry.get_model("qwen3-unsloth-q4ks")
    if not qwen_model:
        print("❌ qwen3-unsloth-q4ks model not found in registry")
        return False
    
    print(f"🔍 Found model: {qwen_model.name}")
    print(f"   Model ID: {qwen_model.model_id}")
    print(f"   Engine: {qwen_model.engine_type}")
    print(f"   Filename: {qwen_model.model_filename}")
    
    # Test model factory creation
    try:
        print(f"🏭 Creating model via ModelFactory...")
        model_instance = ModelFactory.create_model(qwen_model)
        print(f"✅ Model created successfully: {type(model_instance).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Testing GGUF Model Loading Fixes")
    
    success = test_gguf_model_creation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! GGUF model loading should work now.")
    else:
        print("💥 Some tests failed. Check the output above for details.")
    print("=" * 60)

if __name__ == "__main__":
    main()
