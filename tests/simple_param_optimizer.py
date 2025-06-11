#!/usr/bin/env python3
"""
Simplified Parameter Optimization Script
Tests parameter combinations with your existing CLI to find optimal settings.
"""
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ParamResult:
    """Result from testing a parameter combination."""
    params: Dict[str, Any]
    tokens_per_sec: float
    response_length: int
    response: str
    quality_score: float

class SimpleParamOptimizer:
    """Simple parameter optimizer using CLI subprocess calls."""
    
    def __init__(self):
        # Parameter search space
        self.SEARCH_SPACE = {
            'temperature': [0.1, 0.3, 0.5, 0.7, 0.9],
            'top_p': [0.8, 0.85, 0.9, 0.95, 1.0],
            'top_k': [10, 20, 40, 80],
            'repetition_penalty': [1.0, 1.05, 1.1, 1.15],
            'max_new_tokens': [256, 512, 1024]
        }
        
        # Test prompts
        self.TEST_PROMPTS = [
            "Explain artificial intelligence briefly.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "What is blockchain technology?",
            "How does machine learning work?"
        ]
        
        self.results: List[ParamResult] = []
        self.best_result = None
    
    def generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameter combination."""
        return {
            param: random.choice(values)
            for param, values in self.SEARCH_SPACE.items()
        }
    
    def test_params_with_cli(self, params: Dict[str, Any], model: str = "qwen3-unsloth-q4ks") -> ParamResult:
        """Test parameters using the CLI."""
        prompt = random.choice(self.TEST_PROMPTS)
        
        # Build CLI command
        cmd = [
            "python", "-m", "beautyai_inference.cli.unified_cli", 
            "run", "chat",
            "--model", model,
            "--content-filter", "disabled",
            "--disable-thinking"  # Test without thinking first
        ]
        
        # Add generation parameters
        for param, value in params.items():
            if param == "max_new_tokens":
                cmd.extend(["--max-tokens", str(value)])
            else:
                cmd.extend([f"--{param.replace('_', '-')}", str(value)])
        
        print(f"🧪 Testing: {params}")
        print(f"📝 Prompt: {prompt[:50]}...")
        
        try:
            # Run the command
            start_time = time.time()
            
            # Use subprocess to send prompt and get response
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/home/lumi/beautyai"
            )
            
            # Send prompt and exit command
            stdout, stderr = process.communicate(input=f"{prompt}\nexit\n", timeout=60)
            
            total_time = time.time() - start_time
            
            # Parse output to extract response and timing
            response = self.parse_cli_output(stdout)
            response_length = len(response.split())
            tokens_per_sec = response_length / total_time if total_time > 0 else 0
            quality_score = self.calculate_quality_score(prompt, response)
            
            result = ParamResult(
                params=params,
                tokens_per_sec=tokens_per_sec,
                response_length=response_length,
                response=response,
                quality_score=quality_score
            )
            
            print(f"⚡ Speed: {tokens_per_sec:.1f} tok/s, Quality: {quality_score:.1f}/10")
            return result
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - parameters too slow")
            return ParamResult(params, 0.0, 0, "TIMEOUT", 0.0)
        except Exception as e:
            print(f"❌ Error: {e}")
            return ParamResult(params, 0.0, 0, "ERROR", 0.0)
    
    def parse_cli_output(self, output: str) -> str:
        """Extract model response from CLI output."""
        lines = output.split('\n')
        response_lines = []
        in_response = False
        
        for line in lines:
            if "🤖 Model:" in line:
                in_response = True
                # Extract response after "🤖 Model:"
                response_start = line.find("🤖 Model:") + len("🤖 Model:")
                if response_start < len(line):
                    response_lines.append(line[response_start:].strip())
            elif in_response and line.strip() and not line.startswith('[Generated') and not line.startswith('👤'):
                response_lines.append(line.strip())
            elif line.startswith('👤') or line.startswith('[Generated'):
                in_response = False
        
        return ' '.join(response_lines).strip()
    
    def calculate_quality_score(self, prompt: str, response: str) -> float:
        """Simple quality scoring (0-10)."""
        if not response or response in ["TIMEOUT", "ERROR"]:
            return 0.0
        
        score = 5.0  # Base score
        
        # Length appropriateness
        word_count = len(response.split())
        if 30 <= word_count <= 150:
            score += 2.0
        elif word_count < 10:
            score -= 3.0
        elif word_count > 300:
            score -= 1.0
        
        # Basic coherence
        if response.count('.') >= 2:
            score += 1.0
        if not any(word in response.lower() for word in ['error', 'sorry', 'cannot']):
            score += 1.0
        
        # Relevance check
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        if overlap >= 2:
            score += 1.0
        
        return max(0.0, min(10.0, score))
    
    def run_optimization(self, n_trials: int = 10, model: str = "qwen3-unsloth-q4ks"):
        """Run parameter optimization."""
        print(f"🚀 Starting optimization with {n_trials} trials")
        print(f"🎯 Model: {model}")
        
        best_score = 0.0
        
        for trial in range(n_trials):
            print(f"\n🔄 Trial {trial + 1}/{n_trials}")
            
            # Generate random parameters
            params = self.generate_random_params()
            
            # Test parameters
            result = self.test_params_with_cli(params, model)
            self.results.append(result)
            
            # Track best result
            efficiency_score = result.tokens_per_sec * result.quality_score
            if efficiency_score > best_score:
                best_score = efficiency_score
                self.best_result = result
                print(f"🏆 NEW BEST! Efficiency: {efficiency_score:.2f}")
        
        self.print_results()
    
    def print_results(self):
        """Print optimization results."""
        if not self.results:
            print("❌ No results to display")
            return
        
        print("\n" + "=" * 60)
        print("📊 PARAMETER OPTIMIZATION RESULTS")
        print("=" * 60)
        
        # Sort by efficiency score
        sorted_results = sorted(
            self.results, 
            key=lambda x: x.tokens_per_sec * x.quality_score, 
            reverse=True
        )
        
        print(f"🏆 BEST CONFIGURATION:")
        best = sorted_results[0]
        print(f"   Efficiency Score: {best.tokens_per_sec * best.quality_score:.2f}")
        print(f"   Speed: {best.tokens_per_sec:.1f} tokens/sec")
        print(f"   Quality: {best.quality_score:.1f}/10")
        print(f"   Parameters: {best.params}")
        
        print(f"\n📈 TOP 5 RESULTS:")
        for i, result in enumerate(sorted_results[:5], 1):
            efficiency = result.tokens_per_sec * result.quality_score
            print(f"{i}. Efficiency: {efficiency:.2f} | Speed: {result.tokens_per_sec:.1f} | Quality: {result.quality_score:.1f}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"param_optimization_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'best_params': self.best_result.params if self.best_result else None,
            'results': [
                {
                    'params': r.params,
                    'tokens_per_sec': r.tokens_per_sec,
                    'quality_score': r.quality_score,
                    'efficiency_score': r.tokens_per_sec * r.quality_score
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")

def main():
    """Main function."""
    print("🚀 Simple Parameter Optimizer for BeautyAI")
    print("=" * 50)
    
    # Get user input
    try:
        n_trials = input("Number of trials [10]: ").strip()
        n_trials = int(n_trials) if n_trials else 10
    except ValueError:
        n_trials = 10
    
    try:
        model = input("Model name [qwen3-unsloth-q4ks]: ").strip()
        model = model if model else "qwen3-unsloth-q4ks"
    except:
        model = "qwen3-unsloth-q4ks"
    
    # Run optimization
    optimizer = SimpleParamOptimizer()
    optimizer.run_optimization(n_trials, model)

if __name__ == "__main__":
    main()
