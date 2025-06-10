#!/usr/bin/env python3
"""
Test script to verify model loading with disabled content filtering.
This script will test each model in the registry to see if they load properly.
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from beautyai_inference.config.config_manager import ModelRegistry
from beautyai_inference.services.inference.chat_service import ChatService
from beautyai_inference.core.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if all models in the registry can be loaded properly."""
    
    # Load model registry
    registry_path = os.path.join(os.path.dirname(__file__), '..', '..', 'beautyai_inference', 'config', 'model_registry.json')
    registry = ModelRegistry.load_from_file(registry_path)
    
    if not registry:
        logger.error("❌ Failed to load model registry")
        return False
    
    models = registry.list_models()
    logger.info(f"📋 Found {len(models)} models to test: {', '.join(models)}")
    
    # Create chat service with disabled content filtering
    chat_service = ChatService(content_filter_strictness="disabled")
    model_manager = ModelManager()
    
    successful_models = []
    failed_models = []
    
    for model_name in models:
        logger.info(f"\n🔄 Testing model: {model_name}")
        
        try:
            # Get model config
            model_config = registry.get_model(model_name)
            if not model_config:
                logger.error(f"❌ Could not find config for model: {model_name}")
                failed_models.append(model_name)
                continue
            
            # Try to load the model
            logger.info(f"📥 Loading model {model_name}...")
            model_manager.load_model(model_config)
            logger.info(f"✅ Model {model_name} loaded successfully")
            
            # Try a simple test generation
            logger.info(f"🧪 Testing generation with {model_name}...")
            generation_config = {
                "max_new_tokens": 50,
                "temperature": 0.1,
                "do_sample": True
            }
            
            # Use the model for a simple test
            model = model_manager.get_loaded_model(model_config.model_id)
            if model:
                if hasattr(model, 'generate'):
                    response = model.generate("What is Botox?", **generation_config)
                    logger.info(f"✅ Generation test successful. Response length: {len(response)} chars")
                elif hasattr(model, 'chat'):
                    messages = [{"role": "user", "content": "What is Botox?"}]
                    response = model.chat(messages, max_new_tokens=50)
                    logger.info(f"✅ Chat test successful. Response length: {len(response)} chars")
                else:
                    logger.warning(f"⚠️ Model {model_name} has no generate or chat method")
            
            successful_models.append(model_name)
            
            # Unload model to free memory
            model_manager.unload_model(model_config.model_id)
            logger.info(f"🧹 Model {model_name} unloaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to test model {model_name}: {e}")
            failed_models.append(model_name)
            # Try to unload in case of partial loading
            try:
                model_manager.unload_model(model_config.model_id if model_config else model_name)
            except:
                pass
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 MODEL LOADING TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful models: {len(successful_models)}")
    for model in successful_models:
        logger.info(f"  ✓ {model}")
    
    logger.info(f"\n❌ Failed models: {len(failed_models)}")
    for model in failed_models:
        logger.info(f"  ✗ {model}")
    
    if successful_models:
        logger.info(f"\n🎉 SUCCESS: {len(successful_models)}/{len(models)} models loaded successfully!")
        return True
    else:
        logger.error(f"\n💥 FAILURE: No models could be loaded successfully")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
