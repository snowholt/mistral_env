#!/usr/bin/env python3
"""
Test script to verify thinking mode can be properly disabled.
"""
import requests
import json
import time

def test_thinking_mode_control():
    """Test both enabled and disabled thinking modes."""
    
    base_url = "http://localhost:8000/inference/chat"
    
    test_cases = [
        {
            "name": "Thinking Mode DISABLED",
            "payload": {
                "model_name": "qwen3-unsloth-q4ks",
                "message": "What is 2+2? Please show your work.",
                "thinking_mode": "disable",
                "temperature": 0.1,
                "max_new_tokens": 200
            },
            "expect_thinking": False
        },
        {
            "name": "Thinking Mode ENABLED",
            "payload": {
                "model_name": "qwen3-unsloth-q4ks", 
                "message": "What is 2+2? Please show your work.",
                "thinking_mode": "force",
                "temperature": 0.1,
                "max_new_tokens": 200
            },
            "expect_thinking": True
        },
        {
            "name": "No Think Command",
            "payload": {
                "model_name": "qwen3-unsloth-q4ks",
                "message": "/no_think What is 2+2?",
                "temperature": 0.1,
                "max_new_tokens": 200
            },
            "expect_thinking": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        try:
            print(f"📤 Request payload:")
            print(json.dumps(test_case['payload'], indent=2))
            
            start_time = time.time()
            response = requests.post(base_url, json=test_case['payload'], timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n📊 Response Summary:")
                print(f"  Success: {result.get('success', 'Unknown')}")
                print(f"  Thinking Enabled: {result.get('thinking_enabled', 'Unknown')}")
                print(f"  Response Time: {end_time - start_time:.2f}s")
                print(f"  Tokens Generated: {result.get('tokens_generated', 'Unknown')}")
                print(f"  Tokens/sec: {result.get('tokens_per_second', 'Unknown')}")
                
                response_text = result.get('response', '')
                print(f"\n📝 Response Text ({len(response_text)} chars):")
                print(f"  {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
                
                # Check for thinking content
                has_thinking_tags = '<think>' in response_text or '<thinking>' in response_text
                thinking_content = result.get('thinking_content')
                
                print(f"\n🧠 Thinking Analysis:")
                print(f"  Has <think> tags: {has_thinking_tags}")
                print(f"  Thinking content: {'Yes' if thinking_content else 'No'}")
                print(f"  Expected thinking: {test_case['expect_thinking']}")
                
                # Validate results
                if test_case['expect_thinking']:
                    if thinking_content or has_thinking_tags:
                        print(f"  ✅ PASS: Thinking mode worked as expected")
                    else:
                        print(f"  ❌ FAIL: Expected thinking but none found")
                else:
                    if not thinking_content and not has_thinking_tags:
                        print(f"  ✅ PASS: Thinking disabled as expected")
                    else:
                        print(f"  ❌ FAIL: Expected no thinking but found some")
                        
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n{'='*60}")
    print("🏁 Testing Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_thinking_mode_control()
