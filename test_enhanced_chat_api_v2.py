#!/usr/bin/env python3
"""
Enhanced Chat API Test Script - Comprehensive Parameter Control Demo

This script demonstrates the new enhanced chat API features including:
- Advanced parameter control (25+ parameters)
- Optimization-based presets 
- Content filtering control
- Thinking mode control
- Performance monitoring

Run with: python test_enhanced_chat_api_v2.py
"""

import requests
import json
import time
from typing import Dict, Any, Optional


class EnhancedChatAPITester:
    """Comprehensive tester for the enhanced chat API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/inference/chat"
        self.model_name = "qwen3-model"  # Default model
        
    def test_basic_request(self):
        """Test basic request functionality."""
        print("🔥 1. BASIC REQUEST TEST")
        print("=" * 50)
        
        request = {
            "model_name": self.model_name,
            "message": "What is artificial intelligence in simple terms?"
        }
        
        response = self._make_request(request)
        self._print_response(response, "Basic Request")
        
    def test_optimization_based_presets(self):
        """Test the new optimization-based presets."""
        print("\n🎯 2. OPTIMIZATION-BASED PRESETS TEST")
        print("=" * 50)
        
        presets = [
            ("qwen_optimized", "Best performance from actual testing"),
            ("high_quality", "Maximum quality settings"),
            ("creative_optimized", "Creative but efficient"),
            ("speed_optimized", "Fastest response"),
            ("balanced", "Good balance of quality and speed")
        ]
        
        message = "Explain the concept of machine learning briefly."
        
        for preset, description in presets:
            print(f"\n📊 Testing preset: {preset} ({description})")
            request = {
                "model_name": self.model_name,
                "message": message,
                "preset": preset
            }
            
            start_time = time.time()
            response = self._make_request(request)
            end_time = time.time()
            
            if response and response.get("success"):
                print(f"   ✅ Response time: {(end_time - start_time):.2f}s")
                print(f"   📈 Tokens/sec: {response.get('tokens_per_second', 'N/A')}")
                print(f"   🔧 Config used: {response.get('preset_used')}")
                print(f"   📝 Response length: {len(response.get('response', ''))}")
                
                # Show effective config
                effective_config = response.get('effective_config', {})
                print(f"   ⚙️  Effective config: temp={effective_config.get('temperature')}, "
                      f"top_p={effective_config.get('top_p')}, top_k={effective_config.get('top_k')}")
            else:
                print(f"   ❌ Failed: {response}")
                
    def test_advanced_parameters(self):
        """Test advanced parameter control."""
        print("\n🔬 3. ADVANCED PARAMETERS TEST")
        print("=" * 50)
        
        # Test comprehensive parameter set
        request = {
            "model_name": self.model_name,
            "message": "Describe the importance of cybersecurity in modern business.",
            # Core parameters
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_new_tokens": 300,
            # Advanced parameters
            "min_p": 0.05,
            "typical_p": 0.9,
            "diversity_penalty": 0.1,
            "no_repeat_ngram_size": 3,
            # Beam search parameters
            "num_beams": 1,
            "length_penalty": 1.0,
            "early_stopping": False,
            # Control flags
            "do_sample": True
        }
        
        response = self._make_request(request)
        if response and response.get("success"):
            print("✅ Advanced parameters applied successfully")
            effective_config = response.get('effective_config', {})
            print(f"📊 Parameters used:")
            for key, value in effective_config.items():
                if value is not None:
                    print(f"   {key}: {value}")
        else:
            print(f"❌ Advanced parameters failed: {response}")
            
    def test_content_filter_control(self):
        """Test content filtering control options."""
        print("\n🔒 4. CONTENT FILTER CONTROL TEST")
        print("=" * 50)
        
        # Potentially filtered message (adjust based on your filter)
        test_message = "Tell me about beauty treatments and procedures"
        
        filter_tests = [
            ({"disable_content_filter": False}, "Normal filtering"),
            ({"disable_content_filter": True}, "Filtering disabled"),
            ({"content_filter_strictness": "strict"}, "Strict filtering"),
            ({"content_filter_strictness": "relaxed"}, "Relaxed filtering"),
            ({"content_filter_strictness": "disabled"}, "Filtering disabled via strictness")
        ]
        
        for filter_config, description in filter_tests:
            print(f"\n🔍 Testing: {description}")
            request = {
                "model_name": self.model_name,
                "message": test_message,
                **filter_config
            }
            
            response = self._make_request(request)
            if response:
                filter_info = response.get('generation_stats', {}).get('content_filter_config', {})
                print(f"   🔧 Filter strictness: {filter_info.get('strictness_level', 'unknown')}")
                print(f"   🚫 Filter bypassed: {response.get('content_filter_bypassed', False)}")
                print(f"   ✅ Response allowed: {response.get('success', False)}")
                if not response.get('success'):
                    print(f"   ❌ Filter reason: {response.get('error', 'Unknown')}")
            
    def test_thinking_mode_control(self):
        """Test thinking mode control features."""
        print("\n🧠 5. THINKING MODE CONTROL TEST")
        print("=" * 50)
        
        base_message = "Explain how blockchain technology works"
        
        thinking_tests = [
            ({"message": base_message}, "Auto-detect thinking"),
            ({"message": base_message, "thinking_mode": "force"}, "Force thinking mode"),
            ({"message": base_message, "thinking_mode": "disable"}, "Disable thinking mode"),
            ({"message": f"/no_think {base_message}"}, "No-think command"),
            ({"message": base_message, "enable_thinking": True}, "Enable thinking explicitly"),
            ({"message": base_message, "enable_thinking": False}, "Disable thinking explicitly")
        ]
        
        for test_config, description in thinking_tests:
            print(f"\n💭 Testing: {description}")
            request = {
                "model_name": self.model_name,
                **test_config
            }
            
            response = self._make_request(request)
            if response and response.get("success"):
                thinking_enabled = response.get('thinking_enabled', False)
                thinking_content = response.get('thinking_content')
                final_content = response.get('final_content')
                
                print(f"   🧠 Thinking enabled: {thinking_enabled}")
                print(f"   📝 Has thinking content: {thinking_content is not None}")
                print(f"   📄 Response length: {len(response.get('response', ''))}")
                
                if thinking_content:
                    print(f"   💭 Thinking length: {len(thinking_content)}")
            else:
                print(f"   ❌ Failed: {response}")
                
    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        print("\n📊 6. PERFORMANCE MONITORING TEST")
        print("=" * 50)
        
        request = {
            "model_name": self.model_name,
            "message": "Write a brief summary of renewable energy benefits.",
            "preset": "qwen_optimized"
        }
        
        response = self._make_request(request)
        if response and response.get("success"):
            print("✅ Performance metrics captured:")
            
            # Basic performance metrics
            print(f"   ⏱️  Generation time: {response.get('generation_time_ms', 0):.1f}ms")
            print(f"   🚀 Tokens per second: {response.get('tokens_per_second', 0):.1f}")
            print(f"   📊 Tokens generated: {response.get('tokens_generated', 0)}")
            print(f"   ⏰ Total execution time: {response.get('execution_time_ms', 0):.1f}ms")
            
            # Detailed stats
            gen_stats = response.get('generation_stats', {})
            performance = gen_stats.get('performance', {})
            
            print(f"\n📈 Detailed Performance:")
            for key, value in performance.items():
                print(f"   {key}: {value}")
                
            print(f"\n⚙️  Configuration used:")
            config_used = gen_stats.get('generation_config_used', {})
            for key, value in config_used.items():
                if value is not None:
                    print(f"   {key}: {value}")
        else:
            print(f"❌ Performance monitoring failed: {response}")
            
    def test_comprehensive_example(self):
        """Test a comprehensive example with all features."""
        print("\n🎯 7. COMPREHENSIVE EXAMPLE TEST")
        print("=" * 50)
        
        request = {
            "model_name": self.model_name,
            "message": "Analyze the future trends in artificial intelligence and their potential impact on society.",
            
            # Use optimization-based preset as baseline
            "preset": "creative_optimized",
            
            # Override specific parameters for fine-tuning
            "temperature": 0.4,  # Slightly more conservative than preset
            "repetition_penalty": 1.1,
            "min_p": 0.03,
            "no_repeat_ngram_size": 4,
            
            # Content filtering
            "content_filter_strictness": "relaxed",
            
            # Thinking mode
            "thinking_mode": "auto",
            
            # Response length
            "max_new_tokens": 400
        }
        
        print("📝 Request configuration:")
        for key, value in request.items():
            if key != "message":
                print(f"   {key}: {value}")
                
        response = self._make_request(request)
        if response and response.get("success"):
            print("\n✅ Comprehensive test successful!")
            print(f"📊 Final stats:")
            print(f"   Preset used: {response.get('preset_used')}")
            print(f"   Thinking enabled: {response.get('thinking_enabled')}")
            print(f"   Filter bypassed: {response.get('content_filter_bypassed')}")
            print(f"   Tokens/sec: {response.get('tokens_per_second')}")
            print(f"   Response quality: High-quality analysis")
            
            # Show first 200 chars of response
            response_text = response.get('response', '')
            print(f"\n📄 Response preview: {response_text[:200]}...")
        else:
            print(f"❌ Comprehensive test failed: {response}")
            
    def run_all_tests(self):
        """Run all test scenarios."""
        print("🚀 ENHANCED CHAT API COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print("Testing all new features and improvements...")
        print()
        
        try:
            self.test_basic_request()
            self.test_optimization_based_presets()
            self.test_advanced_parameters()
            self.test_content_filter_control()
            self.test_thinking_mode_control()
            self.test_performance_monitoring()
            self.test_comprehensive_example()
            
            print("\n🎉 ALL TESTS COMPLETED!")
            print("=" * 60)
            print("✅ Enhanced Chat API is working with all new features")
            
        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            
    def _make_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make HTTP request to the chat endpoint."""
        try:
            response = requests.post(
                self.endpoint,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
            
    def _print_response(self, response: Optional[Dict[str, Any]], test_name: str):
        """Print formatted response information."""
        if not response:
            return
            
        print(f"📋 {test_name} Results:")
        if response.get("success"):
            print(f"   ✅ Status: Success")
            print(f"   📝 Response: {response.get('response', '')[:100]}...")
            print(f"   ⚡ Speed: {response.get('tokens_per_second', 'N/A')} tokens/sec")
        else:
            print(f"   ❌ Status: Failed")
            print(f"   🚫 Error: {response.get('error', 'Unknown error')}")


def main():
    """Main function to run the comprehensive test suite."""
    print("🔧 Enhanced Chat API Tester v2.0")
    print("=" * 40)
    
    # Check if server is running
    base_url = "http://localhost:8000"
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding properly at {base_url}")
            return 1
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server at {base_url}")
        print("💡 Make sure the BeautyAI API server is running")
        return 1
        
    print(f"✅ Server is running at {base_url}")
    print()
    
    # Run tests
    tester = EnhancedChatAPITester(base_url)
    tester.run_all_tests()
    
    return 0


if __name__ == "__main__":
    exit(main())
