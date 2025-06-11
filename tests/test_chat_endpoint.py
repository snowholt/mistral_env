#!/usr/bin/env python3
"""
Test the chat endpoint with Python requests
"""
import requests
import json

def test_chat_endpoint():
    url = "http://localhost:8000/inference/chat"
    
    payload = {
        "model_name": "qwen3-unsloth-q4ks",
        "message": "What is Botox?",
        "disable_content_filter": True,
        "max_new_tokens": 100
    }
    
    try:
        print("Sending request...")
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success')}")
            print(f"Response: {result.get('response', 'No response')}")
            print(f"Tokens generated: {result.get('tokens_generated')}")
            print(f"Generation time: {result.get('generation_time_ms')}ms")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_chat_endpoint()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
