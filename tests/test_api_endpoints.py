#!/usr/bin/env python3
"""
Comprehensive API endpoint testing script for BeautyAI Inference Framework.

This script systematically tests the /models and /inference endpoints, focusing on:
- Model loading/unloading functionality
- Model listing and registry management  
- Chat inference with various parameters
- Error handling and edge cases
- Tokenizer/model compatibility issues (especially GGUF models)
"""
import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    success: bool
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration: Optional[float] = None

class APIEndpointTester:
    """Comprehensive API endpoint testing class."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.results: List[TestResult] = []
        
        # Add auth header if API key provided
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def log_result(self, result: TestResult):
        """Log and store test result."""
        self.results.append(result)
        status = "✅ PASS" if result.success else "❌ FAIL"
        duration_str = f" ({result.duration:.2f}s)" if result.duration else ""
        logger.info(f"{status} {result.test_name}{duration_str}")
        if result.error_message:
            logger.error(f"   Error: {result.error_message}")
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[bool, Dict[str, Any], Optional[str]]:
        """Make HTTP request and return success, data, error."""
        url = f"{self.base_url}{endpoint}"
        try:
            start_time = time.time()
            response = self.session.request(method, url, **kwargs)
            duration = time.time() - start_time
            
            if response.status_code >= 400:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return False, {}, error_msg
            
            try:
                data = response.json()
                data['_test_duration'] = duration
                return True, data, None
            except json.JSONDecodeError:
                return True, {"raw_response": response.text, "_test_duration": duration}, None
                
        except requests.exceptions.RequestException as e:
            return False, {}, str(e)
    
    # ======================
    # Health Check Tests
    # ======================
    
    def test_health_check(self):
        """Test the health check endpoint."""
        success, data, error = self.make_request("GET", "/health")
        self.log_result(TestResult(
            test_name="Health Check",
            success=success,
            response_data=data,
            error_message=error,
            duration=data.get('_test_duration')
        ))
        return success
    
    # ======================
    # Model Endpoint Tests  
    # ======================
    
    def test_list_models(self):
        """Test listing all models in the registry."""
        success, data, error = self.make_request("GET", "/models/")
        self.log_result(TestResult(
            test_name="List Models",
            success=success and 'models' in data,
            response_data=data,
            error_message=error,
            duration=data.get('_test_duration')
        ))
        return success and data.get('models', [])
    
    def test_get_model_details(self, model_name: str):
        """Test getting details of a specific model."""
        success, data, error = self.make_request("GET", f"/models/{model_name}")
        self.log_result(TestResult(
            test_name=f"Get Model Details ({model_name})",
            success=success and 'model' in data,
            response_data=data,
            error_message=error,
            duration=data.get('_test_duration')
        ))
        return success and data.get('model')
    
    def test_load_model(self, model_name: str):
        """Test loading a specific model."""
        payload = {"model_name": model_name}
        success, data, error = self.make_request("POST", "/models/load", json=payload)
        
        # Consider it a success if the model loads OR if it's already loaded
        model_loaded = success and (
            data.get('success', False) or 
            "already loaded" in str(data.get('message', '')).lower()
        )
        
        self.log_result(TestResult(
            test_name=f"Load Model ({model_name})",
            success=model_loaded,
            response_data=data,
            error_message=error if not model_loaded else None,
            duration=data.get('_test_duration')
        ))
        return model_loaded
    
    def test_unload_model(self, model_name: str):
        """Test unloading a specific model."""
        payload = {"model_name": model_name}
        success, data, error = self.make_request("POST", "/models/unload", json=payload)
        self.log_result(TestResult(
            test_name=f"Unload Model ({model_name})",
            success=success and data.get('success', False),
            response_data=data,
            error_message=error,
            duration=data.get('_test_duration')
        ))
        return success and data.get('success', False)
    
    def test_model_status(self, model_name: str):
        """Test getting the status of a specific model."""
        success, data, error = self.make_request("GET", f"/models/{model_name}/status")
        self.log_result(TestResult(
            test_name=f"Model Status ({model_name})",
            success=success and 'status' in data,
            response_data=data,
            error_message=error,
            duration=data.get('_test_duration')
        ))
        return success and data.get('status')
    
    # ======================
    # Inference Endpoint Tests
    # ======================
    
    def test_basic_chat(self, model_name: str, message: str = "Hello! How are you?"):
        """Test basic chat functionality."""
        payload = {
            "model_name": model_name,
            "message": message
        }
        success, data, error = self.make_request("POST", "/inference/chat", json=payload)
        
        chat_success = success and data.get('success', False) and data.get('response')
        
        self.log_result(TestResult(
            test_name=f"Basic Chat ({model_name})",
            success=chat_success,
            response_data=data,
            error_message=error if not chat_success else None,
            duration=data.get('_test_duration')
        ))
        return chat_success
    
    def test_chat_with_parameters(self, model_name: str):
        """Test chat with custom parameters."""
        payload = {
            "model_name": model_name,
            "message": "Explain the concept of artificial intelligence briefly.",
            "temperature": 0.7,
            "max_new_tokens": 150,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1
        }
        success, data, error = self.make_request("POST", "/inference/chat", json=payload)
        
        param_success = success and data.get('success', False) and data.get('response')
        
        self.log_result(TestResult(
            test_name=f"Chat with Parameters ({model_name})",
            success=param_success,
            response_data=data,
            error_message=error if not param_success else None,
            duration=data.get('_test_duration')
        ))
        return param_success
    
    def test_chat_with_preset(self, model_name: str, preset: str = "balanced"):
        """Test chat with optimization preset."""
        payload = {
            "model_name": model_name,
            "message": "What is machine learning?",
            "preset": preset
        }
        success, data, error = self.make_request("POST", "/inference/chat", json=payload)
        
        preset_success = success and data.get('success', False) and data.get('response')
        
        self.log_result(TestResult(
            test_name=f"Chat with Preset '{preset}' ({model_name})",
            success=preset_success,
            response_data=data,
            error_message=error if not preset_success else None,
            duration=data.get('_test_duration')
        ))
        return preset_success
    
    def test_thinking_mode(self, model_name: str):
        """Test thinking mode functionality."""
        payload = {
            "model_name": model_name,
            "message": "Think about this problem: How do neural networks learn?",
            "enable_thinking": True
        }
        success, data, error = self.make_request("POST", "/inference/chat", json=payload)
        
        thinking_success = success and data.get('success', False) and data.get('response')
        
        self.log_result(TestResult(
            test_name=f"Thinking Mode ({model_name})",
            success=thinking_success,
            response_data=data,
            error_message=error if not thinking_success else None,
            duration=data.get('_test_duration')
        ))
        return thinking_success
    
    def test_content_filter_disabled(self, model_name: str):
        """Test chat with content filter disabled."""
        payload = {
            "model_name": model_name,
            "message": "Write a story about technology.",
            "disable_content_filter": True
        }
        success, data, error = self.make_request("POST", "/inference/chat", json=payload)
        
        filter_success = success and data.get('success', False) and data.get('response')
        
        self.log_result(TestResult(
            test_name=f"Content Filter Disabled ({model_name})",
            success=filter_success,
            response_data=data,
            error_message=error if not filter_success else None,
            duration=data.get('_test_duration')
        ))
        return filter_success
    
    # ======================
    # GGUF/Tokenizer Testing
    # ======================
    
    def test_gguf_model_compatibility(self, model_name: str):
        """Comprehensive test for GGUF model loading and inference."""
        logger.info(f"\n🔬 Starting GGUF compatibility test for {model_name}")
        
        # Step 1: Try to load the model
        load_success = self.test_load_model(model_name)
        if not load_success:
            logger.error(f"❌ GGUF model {model_name} failed to load")
            return False
        
        # Step 2: Check model status
        status = self.test_model_status(model_name)
        if status != "loaded":
            logger.warning(f"⚠️  Model {model_name} status: {status}")
        
        # Step 3: Try basic inference
        chat_success = self.test_basic_chat(model_name, "Test message for GGUF model")
        if not chat_success:
            logger.error(f"❌ GGUF model {model_name} failed basic inference")
            return False
        
        # Step 4: Try with parameters
        param_success = self.test_chat_with_parameters(model_name)
        
        # Step 5: Unload to free memory
        self.test_unload_model(model_name)
        
        overall_success = load_success and chat_success
        logger.info(f"🔬 GGUF compatibility test for {model_name}: {'✅ PASS' if overall_success else '❌ FAIL'}")
        return overall_success
    
    # ======================
    # Full Test Suite
    # ======================
    
    def run_comprehensive_tests(self, target_models: Optional[List[str]] = None):
        """Run comprehensive test suite."""
        logger.info("🚀 Starting comprehensive API endpoint testing")
        logger.info("=" * 60)
        
        # 1. Health check
        logger.info("\n📊 Testing Health Endpoint")
        if not self.test_health_check():
            logger.error("❌ Health check failed - API may not be running")
            return False
        
        # 2. Model listing
        logger.info("\n📋 Testing Model Registry")
        models = self.test_list_models()
        if not models:
            logger.error("❌ Could not retrieve model list")
            return False
        
        logger.info(f"📝 Found {len(models)} models in registry")
        
        # 3. Test specific models or all available models
        test_models = target_models if target_models else [model.get('name', model.get('model_id', '')) for model in models[:3]]
        
        logger.info(f"\n🎯 Testing models: {test_models}")
        
        for model_name in test_models:
            logger.info(f"\n🔍 Testing model: {model_name}")
            logger.info("-" * 40)
            
            # Get model details
            model_details = self.test_get_model_details(model_name)
            if model_details:
                engine_type = model_details.get('engine_type', 'unknown')
                logger.info(f"📦 Engine: {engine_type}")
                
                # Special handling for GGUF models
                if engine_type == "llama.cpp":
                    self.test_gguf_model_compatibility(model_name)
                else:
                    # Standard model testing
                    if self.test_load_model(model_name):
                        self.test_basic_chat(model_name)
                        self.test_chat_with_parameters(model_name)
                        self.test_chat_with_preset(model_name)
                        self.test_unload_model(model_name)
        
        # 4. Summary
        self.print_test_summary()
        return True
    
    def print_test_summary(self):
        """Print a summary of all test results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            logger.info(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.success:
                    logger.info(f"   - {result.test_name}: {result.error_message}")
        
        # Save detailed results to file
        self.save_results_to_file()
    
    def save_results_to_file(self):
        """Save detailed test results to a JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"api_test_results_{timestamp}.json"
        
        results_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_url": self.base_url,
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "error_message": r.error_message,
                    "duration": r.duration,
                    "response_data": r.response_data
                }
                for r in self.results
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"📄 Detailed results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BeautyAI API endpoints")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--focus-gguf", action="store_true", help="Focus on GGUF model testing")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = APIEndpointTester(base_url=args.url, api_key=args.api_key)
    
    # Determine test models
    target_models = args.models
    if args.focus_gguf:
        target_models = [
            "qwen3-unsloth-q4ks",
            "qwen3-unsloth-q4km", 
            "qwen3-official-q4km",
            "bee1reason-arabic-q4ks"
        ]
    
    # Run tests
    try:
        success = tester.run_comprehensive_tests(target_models)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Testing interrupted by user")
        tester.print_test_summary()
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Testing failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
