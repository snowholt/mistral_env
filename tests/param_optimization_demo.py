#!/usr/bin/env python3
"""
Parameter Optimization Demo
Shows how to systematically test different parameter combinations.
"""
import json
import random
import time
from typing import Dict, Any, List

class ParameterTester:
    """Demonstrates parameter optimization approach."""
    
    def __init__(self):
        # Your parameter search space
        self.SEARCH_SPACE = {
            'temperature': [0.1, 0.3, 0.5, 0.7, 0.9],
            'top_p': [0.8, 0.85, 0.9, 0.95, 1.0],
            'top_k': [10, 20, 40, 80],
            'repetition_penalty': [1.0, 1.05, 1.1, 1.15],
            'max_new_tokens': [256, 512, 1024]
        }
        
        # Baseline configurations for comparison
        self.BASELINES = {
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
    
    def generate_random_params(self) -> Dict[str, Any]:
        """Generate a random parameter combination."""
        return {
            param: random.choice(values)
            for param, values in self.SEARCH_SPACE.items()
        }
    
    def generate_smart_combinations(self, n: int = 20) -> List[Dict[str, Any]]:
        """Generate smart parameter combinations."""
        combinations = []
        
        # Add baselines
        for name, params in self.BASELINES.items():
            combinations.append({**params, 'source': f'baseline_{name}'})
        
        # Add random combinations
        for _ in range(n - len(self.BASELINES)):
            params = self.generate_random_params()
            params['source'] = 'random'
            combinations.append(params)
        
        return combinations
    
    def estimate_performance(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance based on parameter values."""
        # This is a simplified model - replace with actual testing
        
        # Speed estimation (higher = faster)
        speed_score = 100.0
        
        # Lower temperature = faster
        speed_score += (1.0 - params['temperature']) * 20
        
        # Lower top_p = faster
        speed_score += (1.0 - params['top_p']) * 15
        
        # Lower top_k = faster
        speed_score += (80 - params['top_k']) * 0.5
        
        # Lower max_tokens = faster
        speed_score += (1024 - params['max_new_tokens']) * 0.1
        
        # Quality estimation (higher = better)
        quality_score = 5.0
        
        # Moderate temperature is often best
        temp_penalty = abs(params['temperature'] - 0.5) * 2
        quality_score -= temp_penalty
        
        # High top_p generally better for quality
        quality_score += params['top_p'] * 3
        
        # Moderate top_k often good
        if 20 <= params['top_k'] <= 40:
            quality_score += 1.0
        
        # Repetition penalty helps quality
        quality_score += (params['repetition_penalty'] - 1.0) * 5
        
        # Normalize scores
        speed_score = max(0, min(100, speed_score))
        quality_score = max(0, min(10, quality_score))
        
        return {
            'speed_score': speed_score,
            'quality_score': quality_score,
            'efficiency_score': speed_score * quality_score
        }
    
    def run_optimization_demo(self, n_combinations: int = 20):
        """Run optimization demonstration."""
        print("🚀 Parameter Optimization Demo")
        print("=" * 50)
        
        # Generate combinations to test
        combinations = self.generate_smart_combinations(n_combinations)
        
        results = []
        
        print(f"Testing {len(combinations)} parameter combinations...\n")
        
        for i, params in enumerate(combinations, 1):
            print(f"🧪 Test {i}/{len(combinations)}")
            print(f"   Params: {dict((k,v) for k,v in params.items() if k != 'source')}")
            
            # Estimate performance (replace with actual testing)
            performance = self.estimate_performance(params)
            
            result = {
                'params': params,
                'performance': performance
            }
            results.append(result)
            
            print(f"   Speed: {performance['speed_score']:.1f}, Quality: {performance['quality_score']:.1f}, Efficiency: {performance['efficiency_score']:.1f}")
            print()
        
        # Analyze results
        self.analyze_results(results)
    
    def analyze_results(self, results: List[Dict]):
        """Analyze and display optimization results."""
        print("=" * 60)
        print("📊 OPTIMIZATION RESULTS")
        print("=" * 60)
        
        # Sort by efficiency score
        sorted_results = sorted(results, key=lambda x: x['performance']['efficiency_score'], reverse=True)
        
        # Best overall
        best = sorted_results[0]
        print(f"🏆 BEST OVERALL CONFIGURATION:")
        print(f"   Efficiency Score: {best['performance']['efficiency_score']:.1f}")
        print(f"   Speed Score: {best['performance']['speed_score']:.1f}")
        print(f"   Quality Score: {best['performance']['quality_score']:.1f}")
        print(f"   Source: {best['params'].get('source', 'unknown')}")
        print("   Parameters:")
        for param, value in best['params'].items():
            if param != 'source':
                print(f"     {param}: {value}")
        
        # Top 5 configurations
        print(f"\n📈 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(sorted_results[:5], 1):
            perf = result['performance']
            source = result['params'].get('source', 'unknown')
            print(f"{i}. Efficiency: {perf['efficiency_score']:.1f} | Speed: {perf['speed_score']:.1f} | Quality: {perf['quality_score']:.1f} | Source: {source}")
        
        # Analysis by source
        print(f"\n📊 PERFORMANCE BY SOURCE:")
        sources = {}
        for result in results:
            source = result['params'].get('source', 'unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(result['performance']['efficiency_score'])
        
        for source, scores in sources.items():
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            print(f"   {source}: Avg={avg_score:.1f}, Max={max_score:.1f}, Count={len(scores)}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Best for speed
        speed_best = max(results, key=lambda x: x['performance']['speed_score'])
        print(f"   For MAX SPEED: temperature={speed_best['params']['temperature']}, top_p={speed_best['params']['top_p']}, max_tokens={speed_best['params']['max_new_tokens']}")
        
        # Best for quality
        quality_best = max(results, key=lambda x: x['performance']['quality_score'])
        print(f"   For MAX QUALITY: temperature={quality_best['params']['temperature']}, top_p={quality_best['params']['top_p']}, repetition_penalty={quality_best['params']['repetition_penalty']}")
        
        # Best balanced
        print(f"   For BALANCED: Use the best overall configuration above")
        
        # Save results
        self.save_results(results)
    
    def save_results(self, results: List[Dict]):
        """Save results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"param_optimization_demo_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def print_cli_commands(self, params: Dict[str, Any]):
        """Print CLI commands to test these parameters."""
        print(f"\n🔧 TO TEST THESE PARAMETERS WITH YOUR CLI:")
        
        cmd_parts = [
            "python -m beautyai_inference.cli.unified_cli run chat",
            "--model qwen3-unsloth-q4ks",
            "--content-filter disabled",
            "--disable-thinking"
        ]
        
        for param, value in params.items():
            if param == 'max_new_tokens':
                cmd_parts.append(f"--max-tokens {value}")
            elif param != 'source':
                cmd_parts.append(f"--{param.replace('_', '-')} {value}")
        
        print("   " + " \\\n     ".join(cmd_parts))

def main():
    """Main demonstration."""
    tester = ParameterTester()
    
    try:
        n_tests = input("Number of parameter combinations to test [20]: ").strip()
        n_tests = int(n_tests) if n_tests else 20
    except ValueError:
        n_tests = 20
    
    tester.run_optimization_demo(n_tests)
    
    # Show CLI command for best result
    print(f"\nWould you like to see CLI commands for testing? (y/n): ", end="")
    response = input().strip().lower()
    if response.startswith('y'):
        # This would show CLI commands for the best parameters found
        print("CLI commands would be shown here for manual testing")

if __name__ == "__main__":
    main()
