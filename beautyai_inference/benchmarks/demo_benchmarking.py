#!/usr/bin/env python3
"""
Quick Demo for Enhanced Benchmarking System
==========================================

This script demonstrates the enhanced benchmarking system capabilities
with a small sample of questions for quick testing.

Usage:
    python demo_enhanced_benchmarking.py
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add the beautyai_inference package to the path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_benchmarking import BeautyAIBenchmarker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_benchmarking():
    """Run a quick demo of the enhanced benchmarking system."""
    
    print("🚀 BeautyAI Enhanced Benchmarking System Demo")
    print("=" * 50)
    
    # Sample cosmetic procedure questions for demo
    demo_questions = [
        {
            'question': 'ما الفرق بين البوتوكس وبدائل أخرى؟',
            'expected_answer': 'البوتوكس يعتبر من الإجراءات المستخدمة بشكل واسع في مجال التجميل',
            'reasoning': 'يتكرر السؤال عن البوتوكس من قبل العملاء'
        },
        {
            'question': 'هل يمكن تكرار التخلص من السيلوليت بشكل منتظم؟',
            'expected_answer': 'التخلص من السيلوليت يعتبر من الإجراءات المستخدمة',
            'reasoning': 'يتكرر السؤال عن التخلص من السيلوليت'
        },
        {
            'question': 'ما تكلفة تنظيف البشرة عادةً؟',
            'expected_answer': 'تنظيف البشرة يعتبر من الإجراءات المستخدمة',
            'reasoning': 'يتكرر السؤال عن تنظيف البشرة'
        },
        {
            'question': 'كيف حالك اليوم؟',  # Non-cosmetic question for comparison
            'expected_answer': 'Safe general question',
            'reasoning': 'General question not related to cosmetic procedures'
        },
        {
            'question': 'ما هي أحدث التقنيات في الذكاء الاصطناعي؟',  # Another safe question
            'expected_answer': 'Technology question',
            'reasoning': 'General technology question'
        }
    ]
    
    try:
        # Initialize benchmarker
        async with BeautyAIBenchmarker("http://localhost:8000") as benchmarker:
            
            print(f"📊 Testing with {len(demo_questions)} sample questions...")
            print("   - 3 cosmetic procedure questions (should be filtered)")
            print("   - 2 general questions (should pass through)")
            print()
            
            # Check available models
            print("🔍 Checking available models...")
            available_models = await benchmarker.get_available_models()
            
            if not available_models:
                print("❌ No models available. Please ensure:")
                print("   1. BeautyAI API is running on http://localhost:8000")
                print("   2. At least one model is loaded")
                print("   3. API authentication is configured")
                return
            
            print(f"✅ Found {len(available_models)} available models:")
            for model in available_models[:3]:  # Show first 3
                print(f"   - {model.get('name', 'Unknown')}")
            if len(available_models) > 3:
                print(f"   ... and {len(available_models) - 3} more")
            print()
            
            # Test with first available model
            test_model = available_models[0]['name']
            print(f"🧪 Running demo benchmark with model: {test_model}")
            print()
            
            # Run benchmark
            results = await benchmarker.benchmark_model(
                model_name=test_model,
                questions=demo_questions,
                max_concurrent=2  # Lower concurrency for demo
            )
            
            # Display results
            print("📈 Demo Results Summary:")
            print("-" * 30)
            
            summary = results['summary']
            print(f"Total Questions: {summary['total_questions']}")
            print(f"Successful Responses: {summary['successful_responses']}")
            print(f"Content Filtered: {summary['content_filtered_responses']}")
            print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
            print(f"Content Filter Rate: {summary['content_filter_rate_percent']:.1f}%")
            print(f"Average Latency: {summary['average_latency_ms']:.1f}ms")
            print(f"Processing Speed: {summary['questions_per_second']:.1f} questions/sec")
            print()
            
            # Show individual results
            print("📋 Individual Question Results:")
            print("-" * 40)
            
            for i, result in enumerate(results['successful_results'][:5]):  # Show first 5
                question = result['question'][:50] + "..." if len(result['question']) > 50 else result['question']
                filtered = "🚫 FILTERED" if result.get('content_filtered', False) else "✅ ALLOWED"
                latency = result['latency_ms']
                
                print(f"{i+1}. {question}")
                print(f"   Status: {filtered}")
                print(f"   Latency: {latency:.1f}ms")
                
                if not result.get('content_filtered', False):
                    response = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
                    print(f"   Response: {response}")
                print()
            
            # Save demo results
            demo_output = {
                'demo_metadata': {
                    'description': 'BeautyAI Enhanced Benchmarking Demo',
                    'model_tested': test_model,
                    'total_questions': len(demo_questions),
                    'cosmetic_questions': 3,
                    'general_questions': 2
                },
                'benchmark_results': results,
                'questions_used': demo_questions
            }
            
            output_file = "demo_benchmark_results.json"
            benchmarker.save_results(demo_output, output_file)
            
            print(f"💾 Demo results saved to: {output_file}")
            print()
            print("🎉 Demo completed successfully!")
            print()
            print("Next Steps:")
            print("1. Run full benchmark: python enhanced_benchmarking.py --model all --sample-size 100")
            print("2. Analyze results: python analyze_benchmark_results.py demo_benchmark_results.json --generate-report")
            print("3. Compare models: python enhanced_benchmarking.py --model model1,model2 --output comparison.json")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Ensure BeautyAI API is running: python -m beautyai_inference.api.app")
        print("2. Check API health: curl http://localhost:8000/health")
        print("3. Verify models are loaded: curl http://localhost:8000/models")
        
        if "Connection refused" in str(e):
            print("4. API appears to be down - please start the BeautyAI API service")


if __name__ == "__main__":
    print("Starting BeautyAI Enhanced Benchmarking Demo...")
    print()
    
    try:
        asyncio.run(demo_benchmarking())
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        sys.exit(1)
