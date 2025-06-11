#!/usr/bin/env python3
"""
Test specific chat parameters that might be causing hanging
"""
import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from beautyai_inference.config.config_manager import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_parameters():
    """Test model parameters to see if there's a parameter issue."""
    logger.info("Testing model parameters...")
    
    try:
        # Load model registry
        registry_path = "/home/lumi/beautyai/beautyai_inference/config/model_registry.json"
        registry = ModelRegistry.load_from_file(registry_path)
        model_config = registry.get_model("qwen3-unsloth-q4ks")
        
        if not model_config:
            logger.error("Model config not found!")
            return False
        
        logger.info(f"Model config: {model_config.to_dict()}")
        
        # Check the model file path
        from beautyai_inference.inference_engines.llamacpp_engine import LlamaCppEngine
        
        engine = LlamaCppEngine(model_config)
        logger.info("Created engine instance")
        
        # Try to find the model path without loading
        model_path = engine._find_gguf_model_path()
        logger.info(f"Model path: {model_path}")
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Model file exists: {model_path}")
            file_size = os.path.getsize(model_path) / (1024**3)  # GB
            logger.info(f"Model file size: {file_size:.2f} GB")
        else:
            logger.error(f"Model file not found: {model_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_parameters()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
