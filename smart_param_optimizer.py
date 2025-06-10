#!/usr/bin/env python3
"""
Smart Parameter Optimization Script for BeautyAI
Optimizes generation parameters (temperature, top_p, top_k, etc.) using random search
and early stopping to find the best balance of speed and quality.
"""
import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Results from a parameter combination test."""
    params: Dict[str, Any]
    speed_tokens_per_sec: float
    response_length: int
    response_quality_score: float  # 0-10 (human judgment or automated)
    inference_time: float
    prompt: str
    response: str
    timestamp: str
    
    @property
    def efficiency_score(self) -> float:
        """Combined score: speed * quality"""
        return self.speed_tokens_per_sec * self.response_quality_score

class ParameterOptimizer:
    """Smart parameter optimization using random search and early stopping."""
    
    def __init__(self, model_name: str = "qwen3-unsloth-q4ks"):
        self.model_name = model_name
        self.model = None
        self.results: List[TestResult] = []
        self.best_params = None
        self.best_score = 0.0
        
        # Parameter search space
        self.SEARCH_SPACE = {
            'temperature': [0.1, 0.3, 0.5, 0.7, 0.9],
            'top_p': [0.8, 0.85, 0.9, 0.95, 1.0],
            'top_k': [10, 20, 40, 80],
            'repetition_penalty': [1.0, 1.05, 1.1, 1.15],
            'max_new_tokens': [256, 512, 1024]
        }
        
        # Test prompts for evaluation
        self.TEST_PROMPTS = [
            "Explain the concept of artificial intelligence in simple terms.",
            "Write a brief summary of renewable energy benefits.",
            "What are the main differences between Python and JavaScript?",
            "Describe the process of photosynthesis.",
            "What are some effective time management techniques?",
            "Explain blockchain technology and its applications.",
            "What are the causes and effects of climate change?",
            "Describe the importance of cybersecurity in modern business."
        ]
        
        # Baseline configurations for comparison
        self.BASELINE_CONFIGS = {
            'conservative': {
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 10,
                'repetition_penalty': 1.0,
                'max_new_tokens': 256
            },
            'balanced': {
                'temperature': 0.5,
                'top_p': 0.9,
                'top_k': 20,
                'repetition_penalty': 1.05,
                'max_new_tokens': 512
            },
            'creative': {
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 40,
                'repetition_penalty': 1.1,
                'max_new_tokens': 1024
            }
        }
        
    def load_model(self):
        """Load the model for testing."""
        if self.model is not None:
            return
            
        logger.info(f"🔄 Loading model: {self.model_name}")
        
        try:
            from beautyai_inference.config.config_manager import AppConfig
            from beautyai_inference.services.model.registry_service import RegistryService
            from beautyai_inference.inference_engines.llamacpp_engine import LlamaCppEngine
            
            # Load model config from registry
            registry_path = Path("beautyai_inference/config/model_registry.json")
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            if self.model_name not in registry_data["models"]:
                raise ValueError(f"Model {self.model_name} not found in registry")
            
            model_data = registry_data["models"][self.model_name]
            
            # Create a minimal ModelConfig-like object
            class ModelConfig:
                def __init__(self, data):
                    self.model_id = data["model_id"]
                    self.engine_type = data["engine_type"]
                    self.max_new_tokens = data.get("max_new_tokens", 1024)
                    self.name = data["name"]
                    self.model_filename = data.get("model_filename")
                    self.temperature = data.get("custom_generation_params", {}).get("temperature", 0.7)
            
            model_config = ModelConfig(model_data)
            
            # Initialize the engine
            self.model = LlamaCppEngine(model_config)
            self.model.load_model()
            
            logger.info(f"✅ Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_random_params(self) -> Dict[str, Any]:
        """Generate a random parameter combination."""
        return {
            param: random.choice(values)
            for param, values in self.SEARCH_SPACE.items()
        }
    
    def test_parameter_combination(self, params: Dict[str, Any], prompt: str) -> TestResult:
        """Test a specific parameter combination."""
        logger.info(f"🧪 Testing params: {params}")
        
        start_time = time.time()
        
        try:
            # Generate response
            response = self.model.generate(prompt, **params)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            response_length = len(response.split())
            speed_tokens_per_sec = response_length / inference_time if inference_time > 0 else 0
            
            # Simple quality scoring (can be improved with human evaluation)
            quality_score = self.calculate_quality_score(prompt, response, params)
            
            result = TestResult(
                params=params,
                speed_tokens_per_sec=speed_tokens_per_sec,
                response_length=response_length,
                response_quality_score=quality_score,
                inference_time=inference_time,
                prompt=prompt,
                response=response,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"⚡ Speed: {speed_tokens_per_sec:.1f} tok/s, Quality: {quality_score:.1f}/10, Length: {response_length}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error testing params: {e}")
            # Return a failed result
            return TestResult(
                params=params,
                speed_tokens_per_sec=0.0,
                response_length=0,
                response_quality_score=0.0,
                inference_time=0.0,
                prompt=prompt,
                response="ERROR",
                timestamp=datetime.now().isoformat()
            )
    
    def calculate_quality_score(self, prompt: str, response: str, params: Dict[str, Any]) -> float:
        """Calculate a quality score for the response (0-10)."""
        score = 5.0  # Base score
        
        # Length appropriateness
        if 50 <= len(response.split()) <= 200:
            score += 1.0
        elif len(response.split()) < 20:
            score -= 2.0
        elif len(response.split()) > 500:
            score -= 1.0
        
        # Coherence checks
        if response.count('.') >= 2:  # Has sentences
            score += 0.5
        if not any(word in response.lower() for word in ['error', 'sorry', 'cannot', "can't"]):
            score += 0.5
        
        # Repetition penalty effectiveness
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.8:
                score += 1.0
            elif unique_ratio < 0.5:
                score -= 2.0
        
        # Response relevance (basic keyword matching)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        if overlap >= 2:
            score += 1.0
        
        # Ensure score is in valid range
        return max(0.0, min(10.0, score))
    
    def run_baseline_tests(self) -> None:
        """Test baseline configurations."""
        logger.info("🎯 Running baseline tests...")
        
        for config_name, params in self.BASELINE_CONFIGS.items():
            logger.info(f"📊 Testing {config_name} baseline...")
            
            # Test with a representative prompt
            prompt = random.choice(self.TEST_PROMPTS)
            result = self.test_parameter_combination(params, prompt)
            result.params['config_type'] = f'baseline_{config_name}'
            self.results.append(result)
            
            # Update best if this is better
            if result.efficiency_score > self.best_score:
                self.best_score = result.efficiency_score
                self.best_params = result.params.copy()
                logger.info(f"🏆 New best baseline: {config_name} (score: {result.efficiency_score:.2f})")
    
    def run_random_search(self, n_trials: int = 50) -> None:
        """Run random search optimization."""
        logger.info(f"🎲 Running random search with {n_trials} trials...")
        
        improvements = 0
        trials_since_improvement = 0
        max_trials_without_improvement = 15
        
        for trial in range(n_trials):
            if trials_since_improvement >= max_trials_without_improvement:
                logger.info(f"🛑 Early stopping: No improvement in {max_trials_without_improvement} trials")
                break
                
            params = self.generate_random_params()
            prompt = random.choice(self.TEST_PROMPTS)
            
            logger.info(f"🔄 Trial {trial + 1}/{n_trials}")
            result = self.test_parameter_combination(params, prompt)
            result.params['config_type'] = 'random_search'
            self.results.append(result)
            
            # Check for improvement
            if result.efficiency_score > self.best_score:
                self.best_score = result.efficiency_score
                self.best_params = result.params.copy()
                improvements += 1
                trials_since_improvement = 0
                logger.info(f"🏆 NEW BEST! Score: {result.efficiency_score:.2f} (improvement #{improvements})")
                logger.info(f"   Params: {params}")
            else:
                trials_since_improvement += 1
                
            # Progress update
            if (trial + 1) % 10 == 0:
                logger.info(f"📈 Progress: {trial + 1}/{n_trials} trials, {improvements} improvements found")
    
    def fine_tune_best(self) -> None:
        """Fine-tune the best parameters found."""
        if not self.best_params:
            logger.warning("⚠️ No best parameters to fine-tune")
            return
            
        logger.info("🔧 Fine-tuning best parameters...")
        
        # Test the best params with multiple prompts for robustness
        total_score = 0.0
        test_prompts = random.sample(self.TEST_PROMPTS, min(3, len(self.TEST_PROMPTS)))
        
        for prompt in test_prompts:
            result = self.test_parameter_combination(self.best_params, prompt)
            result.params['config_type'] = 'fine_tuned'
            self.results.append(result)
            total_score += result.efficiency_score
        
        avg_score = total_score / len(test_prompts)
        logger.info(f"🎯 Fine-tuned average score: {avg_score:.2f}")
    
    def save_results(self, filename: str = None) -> None:
        """Save optimization results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"param_optimization_results_{self.model_name}_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        results_data = {
            'model_name': self.model_name,
            'optimization_timestamp': datetime.now().isoformat(),
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_trials': len(self.results),
            'search_space': self.SEARCH_SPACE,
            'results': [asdict(result) for result in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filename}")
    
    def print_summary(self) -> None:
        """Print optimization summary."""
        if not self.results:
            logger.warning("⚠️ No results to summarize")
            return
        
        print("\n" + "=" * 60)
        print("📊 PARAMETER OPTIMIZATION RESULTS")
        print("=" * 60)
        
        print(f"Model: {self.model_name}")
        print(f"Total trials: {len(self.results)}")
        print(f"Best efficiency score: {self.best_score:.2f}")
        
        print(f"\n🏆 BEST PARAMETERS:")
        for param, value in self.best_params.items():
            if param != 'config_type':
                print(f"   {param}: {value}")
        
        # Top 5 results
        sorted_results = sorted(self.results, key=lambda x: x.efficiency_score, reverse=True)
        print(f"\n📈 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i}. Score: {result.efficiency_score:.2f} | Speed: {result.speed_tokens_per_sec:.1f} tok/s | Quality: {result.response_quality_score:.1f}/10")
            print(f"   Params: {result.params}")
        
        # Statistics
        speeds = [r.speed_tokens_per_sec for r in self.results if r.speed_tokens_per_sec > 0]
        qualities = [r.response_quality_score for r in self.results]
        
        if speeds and qualities:
            print(f"\n📊 STATISTICS:")
            print(f"   Average speed: {sum(speeds)/len(speeds):.1f} tokens/sec")
            print(f"   Max speed: {max(speeds):.1f} tokens/sec")
            print(f"   Average quality: {sum(qualities)/len(qualities):.1f}/10")
            print(f"   Max quality: {max(qualities):.1f}/10")

def main():
    """Main optimization routine."""
    print("🚀 BeautyAI Parameter Optimizer")
    print("=" * 50)
    
    # Load available GGUF models from registry
    try:
        registry_path = Path("beautyai_inference/config/model_registry.json")
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Filter GGUF models only
        gguf_models = []
        for name, config in registry["models"].items():
            if config.get("engine_type") == "llama.cpp":
                gguf_models.append((name, config.get("description", name)))
        
        if not gguf_models:
            print("❌ No GGUF models found in registry")
            return 1
            
        print("Available GGUF models:")
        for i, (name, desc) in enumerate(gguf_models, 1):
            print(f"  {i}. {name}")
    
    except Exception as e:
        print(f"❌ Error loading model registry: {e}")
        return 1
    
    # Model selection
    try:
        choice = input(f"\nSelect model (1-{len(gguf_models)}) [1]: ").strip()
        model_idx = int(choice) - 1 if choice else 0
        selected_model = gguf_models[model_idx][0]
    except (ValueError, IndexError):
        selected_model = gguf_models[0][0]
    
    print(f"\n🎯 Optimizing parameters for: {selected_model}")
    
    # Optimization settings
    try:
        n_trials = input("\nNumber of random trials [30]: ").strip()
        n_trials = int(n_trials) if n_trials else 30
    except ValueError:
        n_trials = 30
    
    # Run optimization
    optimizer = ParameterOptimizer(selected_model)
    
    try:
        # Load model
        optimizer.load_model()
        
        # Run optimization phases
        print(f"\n🔄 Starting optimization with {n_trials} trials...")
        
        # Phase 1: Baseline tests
        optimizer.run_baseline_tests()
        
        # Phase 2: Random search
        optimizer.run_random_search(n_trials)
        
        # Phase 3: Fine-tuning
        optimizer.fine_tune_best()
        
        # Results
        optimizer.print_summary()
        optimizer.save_results()
        
        print(f"\n✅ Optimization completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Optimization interrupted by user")
        if optimizer.results:
            optimizer.print_summary()
            optimizer.save_results()
        return 1
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
