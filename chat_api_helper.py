#!/usr/bin/env python3
"""
CLI helper for testing the enhanced chat API endpoint.
Shows before/after examples of the improved interface.
"""
import argparse
import json
import requests
import sys


def show_examples():
    """Show before/after examples."""
    print("🔄 Enhanced Chat API - Before vs After")
    print("=" * 60)
    
    print("\n❌ OLD WAY (Complex JSON):")
    print("-" * 30)
    old_example = {
        "model_name": "qwen3-model",
        "message": "What is AI?",
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "max_new_tokens": 512,
            "do_sample": True
        }
    }
    print(json.dumps(old_example, indent=2))
    
    print("\n✅ NEW WAY (Simple Direct Parameters):")
    print("-" * 30)
    new_example = {
        "model_name": "qwen3-model",
        "message": "What is AI?",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "max_new_tokens": 512
    }
    print(json.dumps(new_example, indent=2))
    
    print("\n🎯 EVEN SIMPLER (Smart Presets):")
    print("-" * 30)
    preset_example = {
        "model_name": "qwen3-model",
        "message": "What is AI?",
        "preset": "balanced"
    }
    print(json.dumps(preset_example, indent=2))
    
    print("\n🧠 THINKING MODE CONTROL:")
    print("-" * 30)
    thinking_examples = [
        {
            "description": "Disable thinking with command",
            "request": {
                "model_name": "qwen3-model",
                "message": "/no_think Explain briefly",
                "preset": "speed_optimized"
            }
        },
        {
            "description": "Force thinking mode",
            "request": {
                "model_name": "qwen3-model", 
                "message": "Solve this complex problem",
                "thinking_mode": "force",
                "preset": "creative"
            }
        }
    ]
    
    for example in thinking_examples:
        print(f"\n{example['description']}:")
        print(json.dumps(example['request'], indent=2))

    print("\n🆕 NEW ADVANCED FEATURES:")
    print("-" * 30)
    advanced_example = {
        "model_name": "qwen3-model",
        "message": "Explain quantum computing",
        "preset": "qwen_optimized",           # Optimization-based preset
        "min_p": 0.05,                       # Advanced sampling
        "no_repeat_ngram_size": 3,          # Repetition control
        "disable_content_filter": True,      # Filter control
        "thinking_mode": "auto"              # Thinking control
    }
    print(json.dumps(advanced_example, indent=2))
    
    print("\n🔒 CONTENT FILTER CONTROL:")
    print("-" * 30)
    filter_example = {
        "model_name": "qwen3-model",
        "message": "Tell me about beauty treatments",
        "content_filter_strictness": "relaxed"  # or "disabled"
    }
    print(json.dumps(filter_example, indent=2))


def test_request(args):
    """Test a request with the enhanced API."""
    base_url = args.url or "http://localhost:8000"
    endpoint = f"{base_url}/inference/chat"
    
    # Build request based on arguments
    request_data = {
        "model_name": args.model,
        "message": args.message
    }
    
    # Add preset if specified
    if args.preset:
        request_data["preset"] = args.preset
    
    # Add direct parameters if specified
    if args.temperature is not None:
        request_data["temperature"] = args.temperature
    if args.top_p is not None:
        request_data["top_p"] = args.top_p
    if args.top_k is not None:
        request_data["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        request_data["repetition_penalty"] = args.repetition_penalty
    if args.max_tokens is not None:
        request_data["max_new_tokens"] = args.max_tokens
    
    # Add advanced parameters if specified
    if hasattr(args, 'min_p') and args.min_p is not None:
        request_data["min_p"] = args.min_p
    if hasattr(args, 'diversity_penalty') and args.diversity_penalty is not None:
        request_data["diversity_penalty"] = args.diversity_penalty
    if hasattr(args, 'no_repeat_ngram_size') and args.no_repeat_ngram_size is not None:
        request_data["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    
    # Add content filter control if specified
    if hasattr(args, 'disable_filter') and args.disable_filter:
        request_data["disable_content_filter"] = True
    if hasattr(args, 'filter_strictness') and args.filter_strictness:
        request_data["content_filter_strictness"] = args.filter_strictness
    
    # Add thinking mode if specified
    if args.thinking_mode:
        request_data["thinking_mode"] = args.thinking_mode
    
    print("🚀 Testing Enhanced Chat API")
    print("=" * 40)
    print(f"Endpoint: {endpoint}")
    print(f"Request:")
    print(json.dumps(request_data, indent=2))
    print("-" * 40)
    
    try:
        import time
        start_time = time.time()
        response = requests.post(endpoint, json=request_data)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ SUCCESS!")
            print(f"⏱️ Total time: {(end_time - start_time) * 1000:.1f}ms")
            print(f"🎯 Tokens/sec: {data.get('tokens_per_second', 'N/A')}")
            print(f"📋 Preset used: {data.get('preset_used', 'None')}")
            print(f"🧠 Thinking enabled: {data.get('thinking_enabled', 'N/A')}")
            print(f"🔧 Effective config:")
            print(json.dumps(data.get('effective_config', {}), indent=2))
            print(f"\n💬 Response:")
            print(data.get('response', 'No response'))
            
            if data.get('thinking_content'):
                print(f"\n💭 Thinking process:")
                print(data['thinking_content'])
                
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the enhanced chat API endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show examples
  python %(prog)s examples
  
  # Test with preset
  python %(prog)s test -m qwen3-model -t "Hello" --preset balanced
  
  # Test with direct parameters
  python %(prog)s test -m qwen3-model -t "What is AI?" --temperature 0.7 --top-p 0.9
  
  # Test thinking mode
  python %(prog)s test -m qwen3-model -t "Solve: 2x+5=15" --thinking-mode force
  
  # Test no-thinking command
  python %(prog)s test -m qwen3-model -t "/no_think Quick answer" --preset speed_optimized
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Show before/after examples')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a request')
    test_parser.add_argument('-m', '--model', default='qwen3-model', help='Model name')
    test_parser.add_argument('-t', '--message', required=True, help='Message to send')
    test_parser.add_argument('--url', help='API base URL')
    
    # Preset parameters
    test_parser.add_argument('--preset', 
                           choices=['conservative', 'balanced', 'creative', 'speed_optimized', 
                                   'qwen_optimized', 'high_quality', 'creative_optimized'],
                           help='Use optimization-based preset')
    
    # Direct parameters
    test_parser.add_argument('--temperature', type=float, help='Temperature (0.0-2.0)')
    test_parser.add_argument('--top-p', type=float, help='Top-p (0.0-1.0)')
    test_parser.add_argument('--top-k', type=int, help='Top-k (1-100)')
    test_parser.add_argument('--repetition-penalty', type=float, help='Repetition penalty (1.0-2.0)')
    test_parser.add_argument('--max-tokens', type=int, help='Maximum new tokens')
    
    # Advanced parameters
    test_parser.add_argument('--min-p', type=float, help='Minimum probability threshold')
    test_parser.add_argument('--diversity-penalty', type=float, help='Diversity penalty')
    test_parser.add_argument('--no-repeat-ngram-size', type=int, help='N-gram repetition avoidance')
    
    # Content filtering control
    test_parser.add_argument('--disable-filter', action='store_true', help='Disable content filtering')
    test_parser.add_argument('--filter-strictness', 
                           choices=['strict', 'balanced', 'relaxed', 'disabled'],
                           help='Content filter strictness level')
    
    # Advanced parameters
    test_parser.add_argument('--min-p', type=float, help='Minimum probability mass')
    test_parser.add_argument('--diversity-penalty', type=float, help='Diversity penalty')
    test_parser.add_argument('--no-repeat-ngram-size', type=int, help='No repeat n-gram size')
    
    # Content filter control
    test_parser.add_argument('--disable-filter', action='store_true', help='Disable content filter')
    test_parser.add_argument('--filter-strictness', choices=['relaxed', 'strict', 'disabled'],
                           help='Content filter strictness')
    
    # Thinking mode
    test_parser.add_argument('--thinking-mode', choices=['auto', 'force', 'disable'],
                           help='Thinking mode control')
    
    args = parser.parse_args()
    
    if args.command == 'examples':
        show_examples()
    elif args.command == 'test':
        test_request(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
