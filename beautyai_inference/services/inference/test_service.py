"""
Test service for simple model testing functionality.

This service handles basic model testing operations including:
- Running simple prompts against models
- Performance timing
- Response quality validation
- Basic model functionality verification
"""
import logging
import time
from typing import Dict, Any

from ..base.base_service import BaseService
from ...config.config_manager import ModelConfig
from ...core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class TestService(BaseService):
    """Service for simple model testing operations."""
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
    
    def run_test(self, model_name: str, model_config: ModelConfig, 
                 prompt: str, generation_config: Dict[str, Any]) -> int:
        """
        Run a simple test with the model.
        
        Args:
            model_name: Name of the model to test
            model_config: Model configuration
            prompt: Test prompt
            generation_config: Generation parameters
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            print(f"\n=== Testing {model_name} ===")
            print(f"Model ID: {model_config.model_id}")
            print(f"Engine: {model_config.engine_type}")
            print(f"Quantization: {model_config.quantization or 'none'}")
            print(f"Generation parameters: {generation_config}")
            
            # Ensure the model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return 1
            
            # Run the test
            print("\n=== Test Prompt ===")
            print(prompt)
            print("\n=== Response ===")
            
            start_time = time.time()
            
            # Create a minimal context for stateless models
            chat_history = []
            response = model.generate(prompt, generation_config, chat_history)
            
            end_time = time.time()
            
            print(response)
            
            generation_time = end_time - start_time
            tokens_generated = len(response.split())
            
            print(f"\n=== Performance ===")
            print(f"Generation time: {generation_time:.2f}s")
            print(f"Tokens generated: ~{tokens_generated}")
            print(f"Tokens per second: ~{tokens_generated/generation_time:.2f}")
            
            return 0
            
        except Exception as e:
            return self._handle_error(e, f"Failed to test model {model_name}")
    
    def validate_model_response(self, response: str, expected_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a model response against expected criteria.
        
        Args:
            response: The model's response text
            expected_criteria: Validation criteria
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            "valid": True,
            "checks": {},
            "errors": []
        }
        
        try:
            # Check minimum length
            min_length = expected_criteria.get("min_length", 0)
            if len(response) < min_length:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Response too short: {len(response)} < {min_length}")
            validation_results["checks"]["length"] = len(response)
            
            # Check maximum length
            max_length = expected_criteria.get("max_length", float('inf'))
            if len(response) > max_length:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Response too long: {len(response)} > {max_length}")
            
            # Check for required keywords
            required_keywords = expected_criteria.get("required_keywords", [])
            missing_keywords = []
            for keyword in required_keywords:
                if keyword.lower() not in response.lower():
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required keywords: {missing_keywords}")
            validation_results["checks"]["keywords_found"] = len(required_keywords) - len(missing_keywords)
            
            # Check for forbidden content
            forbidden_content = expected_criteria.get("forbidden_content", [])
            found_forbidden = []
            for forbidden in forbidden_content:
                if forbidden.lower() in response.lower():
                    found_forbidden.append(forbidden)
            
            if found_forbidden:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Contains forbidden content: {found_forbidden}")
            validation_results["checks"]["forbidden_content_found"] = len(found_forbidden)
            
            # Check response coherence (basic check)
            if expected_criteria.get("check_coherence", False):
                # Simple coherence check - response should have reasonable sentence structure
                sentences = response.split('.')
                if len(sentences) < 2 and len(response) > 50:
                    validation_results["errors"].append("Response lacks proper sentence structure")
                    validation_results["valid"] = False
                validation_results["checks"]["sentence_count"] = len(sentences)
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def run_multiple_tests(self, model_name: str, model_config: ModelConfig,
                          test_cases: list, generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run multiple test cases against a model.
        
        Args:
            model_name: Name of the model to test
            model_config: Model configuration
            test_cases: List of test case dictionaries
            generation_config: Generation parameters
            
        Returns:
            Dict containing test results
        """
        try:
            print(f"\n=== Running Multiple Tests for {model_name} ===")
            
            # Ensure the model is loaded
            model = self._ensure_model_loaded(model_name, model_config)
            if model is None:
                return {"success": False, "error": "Failed to load model"}
            
            results = {
                "model_name": model_name,
                "total_tests": len(test_cases),
                "passed": 0,
                "failed": 0,
                "test_results": [],
                "summary": {}
            }
            
            total_time = 0
            total_tokens = 0
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n--- Test Case {i}/{len(test_cases)} ---")
                prompt = test_case.get("prompt", "")
                expected_criteria = test_case.get("validation", {})
                test_name = test_case.get("name", f"Test_{i}")
                
                print(f"Test: {test_name}")
                print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
                
                try:
                    # Run the test
                    start_time = time.time()
                    chat_history = []
                    response = model.generate(prompt, generation_config, chat_history)
                    end_time = time.time()
                    
                    generation_time = end_time - start_time
                    tokens_generated = len(response.split())
                    
                    # Validate response if criteria provided
                    validation_result = None
                    if expected_criteria:
                        validation_result = self.validate_model_response(response, expected_criteria)
                    
                    test_result = {
                        "test_name": test_name,
                        "prompt": prompt,
                        "response": response,
                        "generation_time": generation_time,
                        "tokens_generated": tokens_generated,
                        "tokens_per_second": tokens_generated / generation_time,
                        "validation": validation_result,
                        "success": validation_result["valid"] if validation_result else True
                    }
                    
                    results["test_results"].append(test_result)
                    
                    if test_result["success"]:
                        results["passed"] += 1
                        print("✅ PASSED")
                    else:
                        results["failed"] += 1
                        print("❌ FAILED")
                        if validation_result:
                            for error in validation_result["errors"]:
                                print(f"   - {error}")
                    
                    print(f"Response time: {generation_time:.2f}s")
                    print(f"Tokens/sec: {tokens_generated/generation_time:.2f}")
                    
                    total_time += generation_time
                    total_tokens += tokens_generated
                    
                except Exception as e:
                    print(f"❌ ERROR: {str(e)}")
                    results["failed"] += 1
                    results["test_results"].append({
                        "test_name": test_name,
                        "prompt": prompt,
                        "error": str(e),
                        "success": False
                    })
            
            # Calculate summary statistics
            results["summary"] = {
                "success_rate": results["passed"] / results["total_tests"] * 100,
                "total_time": total_time,
                "average_time_per_test": total_time / results["total_tests"],
                "total_tokens": total_tokens,
                "average_tokens_per_test": total_tokens / results["total_tests"],
                "average_tokens_per_second": total_tokens / total_time if total_time > 0 else 0
            }
            
            print(f"\n=== Test Summary ===")
            print(f"Total tests: {results['total_tests']}")
            print(f"Passed: {results['passed']}")
            print(f"Failed: {results['failed']}")
            print(f"Success rate: {results['summary']['success_rate']:.1f}%")
            print(f"Average time per test: {results['summary']['average_time_per_test']:.2f}s")
            print(f"Average tokens per second: {results['summary']['average_tokens_per_second']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running multiple tests: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _ensure_model_loaded(self, model_name: str, model_config: ModelConfig):
        """
        Ensure the model is loaded and return it.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            
        Returns:
            The loaded model
        """
        if not self.model_manager.is_model_loaded(model_name):
            print(f"Loading model '{model_name}'...")
            self.model_manager.load_model(model_name, model_config)
            print(f"Model loaded successfully.")
        else:
            print(f"Using already loaded model '{model_name}'.")
            
        return self.model_manager.get_model(model_name)
