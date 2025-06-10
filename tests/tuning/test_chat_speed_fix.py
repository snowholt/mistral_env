#!/usr/bin/env python3
"""
Test script to verify that the chat interface speed issue is fixed.

This script tests both the optimized LlamaCpp engine directly and 
through the chat service to ensure they have similar performance.
"""

import time
import logging
from beautyai_inference.config.config_manager import ModelConfig
from beautyai_inference.inference_engines.llamacpp_engine import LlamaCppEngine
from beautyai_inference.services.inference.chat_service import ChatService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_engine_directly():
    """Test the LlamaCpp engine directly (like the benchmark)."""
    print("\n" + "="*60)
    print("🧪 TESTING LLAMACPP ENGINE DIRECTLY")
    print("="*60)
    
    # Same config as the benchmark script
    model_config = ModelConfig(
        model_id="mradermacher/Bee1reason-arabic-Qwen-14B-i1-GGUF",
        engine_type="llama.cpp",
        model_filename="Bee1reason-arabic-Qwen-14B.i1-Q4_K_S.gguf",
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.8,
        name="bee1reason-test"
    )
    
    engine = LlamaCppEngine(model_config)
    print("Loading model...")
    engine.load_model()
    print("✅ Model loaded")
    
    # Test prompt
    prompt = "What is artificial intelligence?"
    print(f"Prompt: {prompt}")
    
    # Test using engine.generate() (like benchmark)
    print("\n📊 Testing engine.generate() method:")
    start_time = time.time()
    response = engine.generate(prompt, max_new_tokens=100)
    end_time = time.time()
    
    generation_time = end_time - start_time
    tokens_generated = len(response.split())
    tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
    
    print(f"Response: {response[:100]}...")
    print(f"⚡ Generation time: {generation_time:.2f}s")
    print(f"🎯 Tokens per second: {tokens_per_sec:.1f}")
    print(f"📊 Tokens generated: {tokens_generated}")
    
    # Test using engine.chat() method
    print("\n📊 Testing engine.chat() method:")
    messages = [{"role": "user", "content": prompt}]
    start_time = time.time()
    response = engine.chat(messages, max_new_tokens=100)
    end_time = time.time()
    
    generation_time = end_time - start_time
    tokens_generated = len(response.split())
    tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
    
    print(f"Response: {response[:100]}...")
    print(f"⚡ Generation time: {generation_time:.2f}s")
    print(f"🎯 Tokens per second: {tokens_per_sec:.1f}")
    print(f"📊 Tokens generated: {tokens_generated}")
    
    engine.unload_model()
    return tokens_per_sec

def test_via_chat_service():
    """Test the same engine through the chat service."""
    print("\n" + "="*60)
    print("🗣️ TESTING VIA CHAT SERVICE")
    print("="*60)
    
    # Force load the LlamaCpp model configuration
    model_config = ModelConfig(
        model_id="mradermacher/Bee1reason-arabic-Qwen-14B-i1-GGUF",
        engine_type="llama.cpp",
        model_filename="Bee1reason-arabic-Qwen-14B.i1-Q4_K_S.gguf",
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.8,
        name="bee1reason-test"
    )
    
    # Generation config matching the optimized parameters
    generation_config = {
        "max_new_tokens": 100,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 10,
        "repeat_penalty": 1.05
    }
    
    chat_service = ChatService()
    
    # Ensure model is loaded
    model = chat_service._ensure_model_loaded("bee1reason-test", model_config)
    if model is None:
        print("❌ Failed to load model through chat service")
        return 0
    
    print("✅ Model loaded through chat service")
    
    # Test the same prompt
    prompt = "What is artificial intelligence?"
    chat_history = [{"role": "user", "content": prompt}]
    
    print(f"Prompt: {prompt}")
    print("📊 Testing through chat service:")
    
    start_time = time.time()
    response = model.chat(chat_history, **generation_config)
    end_time = time.time()
    
    generation_time = end_time - start_time
    tokens_generated = len(response.split())
    tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
    
    print(f"Response: {response[:100]}...")
    print(f"⚡ Generation time: {generation_time:.2f}s")
    print(f"🎯 Tokens per second: {tokens_per_sec:.1f}")
    print(f"📊 Tokens generated: {tokens_generated}")
    
    return tokens_per_sec

def main():
    """Main test function."""
    print("🔬 LLAMACPP ENGINE VS CHAT SERVICE SPEED TEST")
    print("Testing the fix for chat interface speed issue")
    print("Expected: Both should achieve similar performance (50+ tokens/sec)")
    
    try:
        # Test engine directly
        engine_speed = test_engine_directly()
        
        # Test via chat service
        chat_speed = test_via_chat_service()
        
        # Compare results
        print("\n" + "="*60)
        print("📊 FINAL COMPARISON")
        print("="*60)
        print(f"🔧 Direct engine speed: {engine_speed:.1f} tokens/sec")
        print(f"💬 Chat service speed:  {chat_speed:.1f} tokens/sec")
        
        if chat_speed > 0:
            speed_ratio = chat_speed / engine_speed if engine_speed > 0 else 0
            print(f"📈 Speed ratio (chat/engine): {speed_ratio:.2f}")
            
            if speed_ratio >= 0.8:  # Within 20% is good
                print("✅ SUCCESS: Chat service performance is comparable to direct engine!")
                if chat_speed >= 50:
                    print("🏆 EXCELLENT: Achieved target speed of 50+ tokens/sec")
                else:
                    print("⚠️ Performance is consistent but below target speed")
            else:
                print("❌ ISSUE: Chat service is significantly slower than direct engine")
                print(f"   Expected similar speeds, got {speed_ratio*100:.1f}% of engine speed")
        else:
            print("❌ FAILED: Chat service did not generate any tokens")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
