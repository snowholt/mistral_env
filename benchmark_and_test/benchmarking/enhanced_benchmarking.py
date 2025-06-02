#!/usr/bin/env python3
"""
Enhanced Benchmarking System for BeautyAI
==========================================

This script creates an independent benchmarking system that:
1. Loads cosmetic procedure questions from 2000QAToR.csv
2. Tests models via BeautyAI API endpoints
3. Collects comprehensive data for external evaluation
4. Enables multi-model comparison for accuracy vs speed analysis

Usage:
    python enhanced_benchmarking.py --help
    python enhanced_benchmarking.py --model qwen3-model --sample-size 100
    python enhanced_benchmarking.py --model all --output-file results.json
"""

import argparse
import json
import csv
import time
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BeautyAIBenchmarker:
    """Enhanced benchmarking system for BeautyAI models."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the benchmarker.
        
        Args:
            api_base_url: Base URL for BeautyAI API
        """
        self.api_base_url = api_base_url
        self.session = None
        self.csv_file_path = Path(__file__).parent.parent / "data" / "2000QAToR.csv"
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def load_cosmetic_questions(self, sample_size: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Load cosmetic procedure questions from CSV file.
        
        Args:
            sample_size: Number of questions to sample (None for all)
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Loading questions from {self.csv_file_path}")
        
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        
        questions = []
        
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    questions.append({
                        'question': row.get('السؤال', ''),
                        'expected_answer': row.get('الإجابة', ''),
                        'reasoning': row.get('سبب التفكير', '')
                    })
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
        
        logger.info(f"Loaded {len(questions)} questions from CSV")
        
        # Sample if requested
        if sample_size and sample_size < len(questions):
            questions = questions[:sample_size]
            logger.info(f"Sampled {sample_size} questions for testing")
        
        return questions
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from BeautyAI API.
        
        Returns:
            List of model information dictionaries
        """
        try:
            async with self.session.get(f"{self.api_base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('models', [])
                else:
                    logger.error(f"Failed to get models: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    async def test_model_with_question(self, model_name: str, question: str) -> Dict[str, Any]:
        """
        Test a single question with a specific model via API.
        
        Args:
            model_name: Name of the model to test
            question: Question to ask the model
            
        Returns:
            Dictionary with test results
        """
        start_time = time.time()
        
        payload = {
            "model_name": model_name,
            "message": question,
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        try:
            async with self.session.post(
                f"{self.api_base_url}/inference/chat",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                end_time = time.time()
                latency = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'success': True,
                        'model_name': model_name,
                        'question': question,
                        'response': data.get('data', {}).get('response', ''),
                        'latency_ms': latency * 1000,
                        'timestamp': datetime.now().isoformat(),
                        'content_filtered': not data.get('data', {}).get('success', True),
                        'api_response': data
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'model_name': model_name,
                        'question': question,
                        'error': f"HTTP {response.status}: {error_text}",
                        'latency_ms': latency * 1000,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                'success': False,
                'model_name': model_name,
                'question': question,
                'error': str(e),
                'latency_ms': latency * 1000,
                'timestamp': datetime.now().isoformat()
            }
    
    async def benchmark_model(self, model_name: str, questions: List[Dict[str, str]], 
                            max_concurrent: int = 5) -> Dict[str, Any]:
        """
        Benchmark a model with all questions.
        
        Args:
            model_name: Name of the model to test
            questions: List of questions to test
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with comprehensive benchmark results
        """
        logger.info(f"Starting benchmark for model: {model_name}")
        logger.info(f"Testing {len(questions)} questions with max {max_concurrent} concurrent requests")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def test_with_semaphore(question_data):
            async with semaphore:
                return await self.test_model_with_question(model_name, question_data['question'])
        
        # Execute all tests
        start_time = time.time()
        tasks = [test_with_semaphore(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Process results
        successful_results = []
        failed_results = []
        content_filtered_count = 0
        total_latency = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    'question': questions[i]['question'],
                    'error': str(result),
                    'model_name': model_name
                })
            elif result['success']:
                successful_results.append(result)
                total_latency += result['latency_ms']
                if result.get('content_filtered', False):
                    content_filtered_count += 1
            else:
                failed_results.append(result)
        
        # Calculate statistics
        total_questions = len(questions)
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        success_rate = (successful_count / total_questions * 100) if total_questions > 0 else 0
        content_filter_rate = (content_filtered_count / total_questions * 100) if total_questions > 0 else 0
        avg_latency = (total_latency / successful_count) if successful_count > 0 else 0
        total_duration = end_time - start_time
        
        summary = {
            'model_name': model_name,
            'total_questions': total_questions,
            'successful_responses': successful_count,
            'failed_responses': failed_count,
            'content_filtered_responses': content_filtered_count,
            'success_rate_percent': round(success_rate, 2),
            'content_filter_rate_percent': round(content_filter_rate, 2),
            'average_latency_ms': round(avg_latency, 2),
            'total_duration_seconds': round(total_duration, 2),
            'questions_per_second': round(total_questions / total_duration, 2) if total_duration > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Benchmark completed for {model_name}:")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Content filter rate: {content_filter_rate:.1f}%")
        logger.info(f"  Average latency: {avg_latency:.1f}ms")
        logger.info(f"  Questions per second: {summary['questions_per_second']:.1f}")
        
        return {
            'summary': summary,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'questions_metadata': questions
        }
    
    async def benchmark_multiple_models(self, model_names: List[str], questions: List[Dict[str, str]],
                                      max_concurrent: int = 5) -> Dict[str, Any]:
        """
        Benchmark multiple models and generate comparison data.
        
        Args:
            model_names: List of model names to test
            questions: List of questions to test
            max_concurrent: Maximum concurrent requests per model
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Starting multi-model benchmark with {len(model_names)} models")
        
        results = {}
        overall_start_time = time.time()
        
        for model_name in model_names:
            try:
                model_results = await self.benchmark_model(model_name, questions, max_concurrent)
                results[model_name] = model_results
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_name}: {e}")
                results[model_name] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        overall_end_time = time.time()
        
        # Generate comparison summary
        comparison_summary = {
            'total_models_tested': len(model_names),
            'total_questions': len(questions),
            'total_benchmark_duration_seconds': round(overall_end_time - overall_start_time, 2),
            'timestamp': datetime.now().isoformat(),
            'model_rankings': self._generate_model_rankings(results)
        }
        
        return {
            'comparison_summary': comparison_summary,
            'model_results': results,
            'questions_metadata': questions
        }
    
    def _generate_model_rankings(self, results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate model rankings based on different criteria."""
        
        valid_models = []
        for model_name, model_data in results.items():
            if 'summary' in model_data:
                summary = model_data['summary']
                valid_models.append({
                    'model_name': model_name,
                    'success_rate': summary.get('success_rate_percent', 0),
                    'avg_latency': summary.get('average_latency_ms', float('inf')),
                    'content_filter_rate': summary.get('content_filter_rate_percent', 0),
                    'questions_per_second': summary.get('questions_per_second', 0)
                })
        
        rankings = {}
        
        # Rank by success rate (descending)
        rankings['by_success_rate'] = sorted(
            valid_models, 
            key=lambda x: x['success_rate'], 
            reverse=True
        )
        
        # Rank by speed (ascending latency)
        rankings['by_speed'] = sorted(
            valid_models, 
            key=lambda x: x['avg_latency']
        )
        
        # Rank by content filtering effectiveness (descending)
        rankings['by_content_filtering'] = sorted(
            valid_models, 
            key=lambda x: x['content_filter_rate'], 
            reverse=True
        )
        
        # Rank by throughput (descending)
        rankings['by_throughput'] = sorted(
            valid_models, 
            key=lambda x: x['questions_per_second'], 
            reverse=True
        )
        
        return rankings
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save benchmark results to JSON file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def generate_csv_report(self, results: Dict[str, Any], output_file: str):
        """Generate CSV report from benchmark results."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Flatten results for CSV
            rows = []
            
            for model_name, model_data in results.get('model_results', {}).items():
                if 'successful_results' in model_data:
                    for result in model_data['successful_results']:
                        rows.append({
                            'model_name': result['model_name'],
                            'question': result['question'],
                            'response': result['response'],
                            'latency_ms': result['latency_ms'],
                            'content_filtered': result.get('content_filtered', False),
                            'timestamp': result['timestamp']
                        })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"CSV report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")
            raise


async def main():
    """Main function to run the enhanced benchmarking system."""
    parser = argparse.ArgumentParser(
        description="Enhanced Benchmarking System for BeautyAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single model with 100 questions
  python enhanced_benchmarking.py --model qwen3-model --sample-size 100
  
  # Test all available models with custom output
  python enhanced_benchmarking.py --model all --output results/benchmark_$(date +%Y%m%d).json
  
  # Test specific models with high concurrency
  python enhanced_benchmarking.py --model qwen3-model,mistral-model --concurrent 10
  
  # Generate both JSON and CSV reports
  python enhanced_benchmarking.py --model all --output results.json --csv-report results.csv
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='all',
        help='Model name to test or "all" for all models (comma-separated for multiple)'
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int, 
        help='Number of questions to sample from CSV (default: all questions)'
    )
    
    parser.add_argument(
        '--concurrent', 
        type=int, 
        default=5,
        help='Maximum concurrent requests per model (default: 5)'
    )
    
    parser.add_argument(
        '--api-url', 
        type=str, 
        default='http://localhost:8000',
        help='BeautyAI API base URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        help='Output file for results (default: timestamp-based filename)'
    )
    
    parser.add_argument(
        '--csv-report', 
        type=str,
        help='Generate additional CSV report with detailed results'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize benchmarker
    async with BeautyAIBenchmarker(args.api_url) as benchmarker:
        
        # Load questions
        questions = benchmarker.load_cosmetic_questions(args.sample_size)
        if not questions:
            logger.error("No questions loaded. Exiting.")
            sys.exit(1)
        
        # Determine models to test
        if args.model.lower() == 'all':
            available_models = await benchmarker.get_available_models()
            if not available_models:
                logger.error("No models available. Make sure BeautyAI API is running.")
                sys.exit(1)
            model_names = [model['name'] for model in available_models]
            logger.info(f"Testing all available models: {', '.join(model_names)}")
        else:
            model_names = [name.strip() for name in args.model.split(',')]
            logger.info(f"Testing specified models: {', '.join(model_names)}")
        
        # Run benchmarks
        if len(model_names) == 1:
            # Single model benchmark
            results = await benchmarker.benchmark_model(model_names[0], questions, args.concurrent)
            output_data = {
                'single_model_results': results,
                'benchmark_config': {
                    'model_name': model_names[0],
                    'sample_size': len(questions),
                    'concurrent_requests': args.concurrent,
                    'api_url': args.api_url
                }
            }
        else:
            # Multi-model comparison
            results = await benchmarker.benchmark_multiple_models(model_names, questions, args.concurrent)
            output_data = {
                'multi_model_comparison': results,
                'benchmark_config': {
                    'model_names': model_names,
                    'sample_size': len(questions),
                    'concurrent_requests': args.concurrent,
                    'api_url': args.api_url
                }
            }
        
        # Save results
        benchmarker.save_results(output_data, args.output)
        
        # Generate CSV report if requested
        if args.csv_report:
            if 'multi_model_comparison' in output_data:
                benchmarker.generate_csv_report(output_data['multi_model_comparison'], args.csv_report)
            else:
                # Convert single model results to multi-model format
                single_model_data = {
                    'model_results': {model_names[0]: output_data['single_model_results']}
                }
                benchmarker.generate_csv_report(single_model_data, args.csv_report)
        
        logger.info("Enhanced benchmarking completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
