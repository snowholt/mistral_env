#!/usr/bin/env python3
"""
Test script to debug the chat issue directly
"""
import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from beautyai_inference.core.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_direct_chat():
    """Test the model's chat method directly."""
    logger.info("Testing direct model chat...")
    
    # Get model manager
    model_manager = ModelManager()
    
    # Check loaded models
    loaded_models = model_manager.list_loaded_models()
    logger.info(f"Loaded models: {loaded_models}")
    
    if not loaded_models:
        logger.error("No models loaded!")
        return False
    
    model_name = loaded_models[0]
    logger.info(f"Testing with model: {model_name}")
    
    # Get the model
    model = model_manager.get_loaded_model(model_name)
    if not model:
        logger.error(f"Failed to get model: {model_name}")
        return False
    
    logger.info(f"Got model: {type(model)}")
    
    # Test simple generation
    try:
        # First try the generate method
        logger.info("Calling model.generate()...")
        response = model.generate("Hello", max_new_tokens=10, temperature=0.1)
        logger.info(f"Generate response: {response}")
        
        # Then try chat method
        messages = [{"role": "user", "content": "Hello"}]
        logger.info("Calling model.chat()...")
        
        response = model.chat(messages, max_new_tokens=10, temperature=0.1)
        logger.info(f"Chat response: {response}")
        logger.info(f"Response length: {len(response)}")
        return True
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_chat()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
