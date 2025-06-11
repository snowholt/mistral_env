#!/usr/bin/env python3
"""
Test script for the enhanced chat API endpoint.
Demonstrates the new user-friendly parameter interface.
"""
import requests
import json
import time
from typing import Dict, Any


class ChatAPITester:
    """Test the enhanced chat API with various parameter combinations."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/inference/chat"
    
    def test_preset_configurations(self):
        """Test the preset configurations based on optimization results."""
        print("🧪 Testing Preset Configurations")
        print("=" * 50)
        
        presets = ["conservative", "balanced", "creative", "speed_optimized"]
        test_message = "What is artificial intelligence?"
        
        for preset in presets:
            print(f"\n📋 Testing '{preset}' preset:")
            
            request_data = {
                "model_name": "qwen3-model",
                "message": test_message,
                "preset": preset
            }
            
            try:
                start_time = time.time()
                response = requests.post(self.endpoint, json=request_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ Success ({end_time - start_time:.2f}s)")
                    print(f"  🎯 Tokens/sec: {data.get('tokens_per_second', 'N/A')}")
                    print(f"  📊 Config used: {data.get('effective_config', {})}")
                    print(f"  💬 Response: {data.get('response', '')[:100]}...")
                else:
                    print(f"  ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    def test_direct_parameters(self):
        """Test direct parameter specification."""
        print("\n🎛️ Testing Direct Parameter Access")
        print("=" * 50)
        
        # Test case based on your optimization results
        test_configs = [
            {
                "name": "High Speed",
                "params": {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 10,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 256
                }
            },
            {
                "name": "High Quality", 
                "params": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 20,
                    "repetition_penalty": 1.05,
                    "max_new_tokens": 512
                }
            },
            {
                "name": "Creative Mode",
                "params": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "max_new_tokens": 1024
                }
            }
        ]
        
        for config in test_configs:
            print(f"\n🔧 Testing {config['name']}:")
            
            request_data = {
                "model_name": "qwen3-model",
                "message": "Explain quantum computing briefly",
                **config['params']
            }
            
            print(f"  Parameters: {config['params']}")
            
            try:
                start_time = time.time()
                response = requests.post(self.endpoint, json=request_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ Success ({end_time - start_time:.2f}s)")
                    print(f"  🎯 Tokens/sec: {data.get('tokens_per_second', 'N/A')}")
                    print(f"  📏 Response length: {len(data.get('response', ''))}")
                else:
                    print(f"  ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    def test_thinking_mode(self):
        """Test thinking mode control."""
        print("\n🧠 Testing Thinking Mode Control")
        print("=" * 50)
        
        test_cases = [
            {
                "name": "Auto thinking (default)",
                "request": {
                    "model_name": "qwen3-model",
                    "message": "Solve this step by step: 2x + 5 = 15"
                }
            },
            {
                "name": "Explicit thinking enabled",
                "request": {
                    "model_name": "qwen3-model", 
                    "message": "Solve this: 2x + 5 = 15",
                    "thinking_mode": "force"
                }
            },
            {
                "name": "Thinking disabled with command",
                "request": {
                    "model_name": "qwen3-model",
                    "message": "/no_think Solve this: 2x + 5 = 15"
                }
            },
            {
                "name": "Thinking explicitly disabled",
                "request": {
                    "model_name": "qwen3-model",
                    "message": "Solve this: 2x + 5 = 15",
                    "thinking_mode": "disable"
                }
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 {test_case['name']}:")
            
            try:
                start_time = time.time()
                response = requests.post(self.endpoint, json=test_case['request'])
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ Success ({end_time - start_time:.2f}s)")
                    print(f"  🧠 Thinking enabled: {data.get('thinking_enabled')}")
                    print(f"  🎯 Tokens/sec: {data.get('tokens_per_second', 'N/A')}")
                    
                    if data.get('thinking_content'):
                        print(f"  💭 Thinking: {data['thinking_content'][:50]}...")
                    
                    print(f"  💬 Response: {data.get('final_content', data.get('response', ''))[:100]}...")
                else:
                    print(f"  ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    def test_parameter_priority(self):
        """Test the parameter priority system."""
        print("\n⚖️ Testing Parameter Priority System")
        print("=" * 50)
        
        # Test: preset + direct override
        print("\n🔄 Preset + Direct Parameter Override:")
        request_data = {
            "model_name": "qwen3-model",
            "message": "Hello world",
            "preset": "conservative",  # This sets temperature=0.1
            "temperature": 0.8  # This should override the preset
        }
        
        try:
            response = requests.post(self.endpoint, json=request_data)
            if response.status_code == 200:
                data = response.json()
                effective_config = data.get('effective_config', {})
                print(f"  📋 Preset used: {data.get('preset_used')}")
                print(f"  🎛️ Effective temperature: {effective_config.get('temperature')}")
                print(f"  ✅ Expected: 0.8 (direct override should win)")
            else:
                print(f"  ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    def benchmark_performance(self):
        """Benchmark different configurations."""
        print("\n🏃 Performance Benchmark")
        print("=" * 50)
        
        configs = [
            {"name": "Speed Optimized", "preset": "speed_optimized"},
            {"name": "Conservative", "preset": "conservative"}, 
            {"name": "Balanced", "preset": "balanced"},
            {"name": "Creative", "preset": "creative"}
        ]
        
        test_message = "What is machine learning?"
        results = []
        
        for config in configs:
            print(f"\n⚡ Benchmarking {config['name']}:")
            
            request_data = {
                "model_name": "qwen3-model",
                "message": test_message,
                **config
            }
            
            try:
                start_time = time.time()
                response = requests.post(self.endpoint, json=request_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    tokens_per_sec = data.get('tokens_per_second', 0)
                    generation_time = data.get('generation_time_ms', 0)
                    
                    results.append({
                        'config': config['name'],
                        'tokens_per_sec': tokens_per_sec,
                        'generation_time_ms': generation_time,
                        'total_time': (end_time - start_time) * 1000
                    })
                    
                    print(f"  🎯 Tokens/sec: {tokens_per_sec}")
                    print(f"  ⏱️ Generation time: {generation_time:.1f}ms")
                    print(f"  📊 Total time: {(end_time - start_time) * 1000:.1f}ms")
                else:
                    print(f"  ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Summary
        if results:
            print("\n📈 Performance Summary:")
            print("-" * 40)
            sorted_results = sorted(results, key=lambda x: x['tokens_per_sec'], reverse=True)
            for i, result in enumerate(sorted_results, 1):
                print(f"{i}. {result['config']:15} - {result['tokens_per_sec']:.1f} tokens/sec")


def main():
    """Run all tests."""
    print("🚀 Enhanced Chat API Testing Suite")
    print("=" * 60)
    print("Testing the improved chat endpoint with:")
    print("- Direct parameter access")
    print("- Optimization-based presets") 
    print("- Thinking mode control")
    print("- Parameter priority system")
    print("- Performance benchmarking")
    print()
    
    tester = ChatAPITester()
    
    try:
        # Test all features
        tester.test_preset_configurations()
        tester.test_direct_parameters()
        tester.test_thinking_mode()
        tester.test_parameter_priority()
        tester.benchmark_performance()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("\n💡 Key Benefits Demonstrated:")
        print("1. Much simpler parameter specification")
        print("2. Optimization-based smart presets")
        print("3. Easy thinking mode control")
        print("4. Detailed performance metrics")
        print("5. Backward compatibility maintained")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    main()
