#!/usr/bin/env python3
"""
Test simpler chat parameters to debug the hanging issue
"""
import sys
import os
import logging
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_llama_chat():
    """Test llama.cpp directly with simple parameters."""
    try:
        from llama_cpp import Llama, ChatCompletionMessage
        
        model_path = "/home/lumi/.cache/huggingface/hub/models--unsloth--Qwen3-14B-GGUF/snapshots/a04a82c4739b3ef5fa6da7d10261db2c67dd1985/Qwen3-14B-Q4_K_S.gguf"
        
        logger.info("Loading llama.cpp model directly...")
        
        # Simple loading parameters
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # All on GPU
            n_ctx=512,        # Small context for testing
            n_batch=512,      # Small batch for testing
            verbose=False
        )
        
        logger.info("Model loaded. Testing simple chat...")
        
        # Create simple message
        messages = [
            ChatCompletionMessage(role="user", content="Hello")
        ]
        
        start_time = time.time()
        
        # Simple chat completion
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=20,
            temperature=0.7,
            stream=False
        )
        
        end_time = time.time()
        logger.info(f"Chat completed in {end_time - start_time:.2f} seconds")
        
        if response and 'choices' in response:
            content = response['choices'][0]['message']['content']
            logger.info(f"Response: {content}")
            return True
        else:
            logger.error("No response from model")
            return False
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_llama_chat()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
