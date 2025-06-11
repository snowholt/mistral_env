#!/usr/bin/env python3
"""
Enhanced Chat API CLI Helper v2.0 - Simplified Parameter Control

This CLI tool demonstrates how much easier it is now to control chat parameters
compared to the complex JSON configuration approach.

Usage examples:
  # Basic usage with preset
  python enhanced_chat_cli.py "What is AI?" --preset qwen_optimized
  
  # Advanced parameter control
  python enhanced_chat_cli.py "Explain quantum computing" --temp 0.3 --top-p 0.95 --top-k 20 --no-filter
  
  # Thinking mode control  
  python enhanced_chat_cli.py "/no_think Give a brief answer" --preset speed_optimized
  
  # Content filtering control
  python enhanced_chat_cli.py "Tell me about beauty treatments" --filter-strictness relaxed
"""

import argparse
import requests
import json
import sys
from typing import Dict, Any, Optional


def create_request(args) -> Dict[str, Any]:
    """Create the API request from CLI arguments."""
    request = {
        "model_name": args.model,
        "message": args.message
    }
    
    # Add preset if specified
    if args.preset:
        request["preset"] = args.preset
        
    # Add direct parameters if specified
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.top_p is not None:
        request["top_p"] = args.top_p
    if args.top_k is not None:
        request["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        request["repetition_penalty"] = args.repetition_penalty
    if args.max_tokens is not None:
        request["max_new_tokens"] = args.max_tokens
    if args.min_tokens is not None:
        request["min_new_tokens"] = args.min_tokens
        
    # Advanced parameters
    if args.min_p is not None:
        request["min_p"] = args.min_p
    if args.typical_p is not None:
        request["typical_p"] = args.typical_p
    if args.diversity_penalty is not None:
        request["diversity_penalty"] = args.diversity_penalty
    if args.no_repeat_ngram_size is not None:
        request["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        
    # Beam search parameters
    if args.num_beams is not None:
        request["num_beams"] = args.num_beams
    if args.length_penalty is not None:
        request["length_penalty"] = args.length_penalty
    if args.early_stopping is not None:
        request["early_stopping"] = args.early_stopping
        
    # Content filtering control
    if args.no_filter:
        request["disable_content_filter"] = True
    if args.filter_strictness:
        request["content_filter_strictness"] = args.filter_strictness
        
    # Thinking mode control
    if args.thinking_mode:
        request["thinking_mode"] = args.thinking_mode
    if args.enable_thinking is not None:
        request["enable_thinking"] = args.enable_thinking
        
    # Sampling control
    if args.do_sample is not None:
        request["do_sample"] = args.do_sample
        
    return request


def send_request(request: Dict[str, Any], base_url: str) -> Optional[Dict[str, Any]]:
    """Send the request to the API."""
    try:
        response = requests.post(
            f"{base_url}/inference/chat",
            json=request,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def print_response(response: Dict[str, Any], show_details: bool = False):
    """Print the formatted response."""
    if not response.get("success"):
        print(f"❌ Error: {response.get('error', 'Unknown error')}")
        return
        
    # Main response
    print("\n🤖 Model Response:")
    print("=" * 50)
    print(response.get("response", ""))
    
    if show_details:
        print("\n📊 Performance Metrics:")
        print(f"   ⚡ Speed: {response.get('tokens_per_second', 0):.1f} tokens/sec")
        print(f"   📝 Tokens: {response.get('tokens_generated', 0)}")
        print(f"   ⏱️  Time: {response.get('generation_time_ms', 0):.0f}ms")
        
        print("\n⚙️  Configuration Used:")
        effective_config = response.get('effective_config', {})
        for key, value in effective_config.items():
            if value is not None:
                print(f"   {key}: {value}")
                
        print("\n🔧 Settings:")
        print(f"   Preset: {response.get('preset_used', 'None')}")
        print(f"   Thinking: {response.get('thinking_enabled', False)}")
        print(f"   Filter bypassed: {response.get('content_filter_bypassed', False)}")
        print(f"   Filter strictness: {response.get('content_filter_strictness', 'N/A')}")
        
        # Show thinking content if available
        thinking_content = response.get('thinking_content')
        if thinking_content:
            print(f"\n💭 Thinking Process:")
            print(thinking_content[:200] + "..." if len(thinking_content) > 200 else thinking_content)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Chat API CLI - Easy Parameter Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use optimization-based preset
  %(prog)s "What is AI?" --preset qwen_optimized
  
  # Fine-tune parameters directly
  %(prog)s "Explain quantum physics" --temp 0.3 --top-p 0.95 --top-k 20
  
  # Disable content filtering
  %(prog)s "Tell me about procedures" --no-filter
  
  # Control thinking mode
  %(prog)s "/no_think Brief answer please" --preset speed_optimized
  
  # Advanced sampling control
  %(prog)s "Creative story" --temp 0.7 --min-p 0.05 --diversity-penalty 0.2
  
Available presets: conservative, balanced, creative, speed_optimized, 
                   qwen_optimized, high_quality, creative_optimized
        """
    )
    
    # Required arguments
    parser.add_argument("message", help="Message to send to the model")
    
    # Basic options
    parser.add_argument("--model", "-m", default="qwen3-model", help="Model name to use")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--details", "-d", action="store_true", help="Show detailed response information")
    
    # Preset configurations
    parser.add_argument("--preset", "-p", 
                      choices=["conservative", "balanced", "creative", "speed_optimized", 
                              "qwen_optimized", "high_quality", "creative_optimized"],
                      help="Use optimization-based preset configuration")
    
    # Core generation parameters
    parser.add_argument("--temperature", "--temp", type=float, help="Temperature (0.0-2.0)")
    parser.add_argument("--top-p", type=float, help="Top-p nucleus sampling (0.0-1.0)")
    parser.add_argument("--top-k", type=int, help="Top-k sampling (1-100)")
    parser.add_argument("--repetition-penalty", "--rep-penalty", type=float, help="Repetition penalty (1.0-2.0)")
    parser.add_argument("--max-tokens", type=int, help="Maximum new tokens to generate")
    parser.add_argument("--min-tokens", type=int, help="Minimum new tokens to generate")
    
    # Advanced sampling parameters
    parser.add_argument("--min-p", type=float, help="Minimum probability threshold")
    parser.add_argument("--typical-p", type=float, help="Typical sampling parameter")
    parser.add_argument("--diversity-penalty", type=float, help="Diversity penalty for varied responses")
    parser.add_argument("--no-repeat-ngram-size", type=int, help="N-gram size for repetition avoidance")
    
    # Beam search parameters
    parser.add_argument("--num-beams", type=int, help="Number of beams for beam search")
    parser.add_argument("--length-penalty", type=float, help="Length penalty for beam search")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping in beam search")
    
    # Content filtering control
    parser.add_argument("--no-filter", action="store_true", help="Disable content filtering entirely")
    parser.add_argument("--filter-strictness", 
                      choices=["strict", "balanced", "relaxed", "disabled"],
                      help="Set content filter strictness level")
    
    # Thinking mode control
    parser.add_argument("--thinking-mode", 
                      choices=["auto", "force", "disable"],
                      help="Control thinking mode behavior")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false", help="Disable thinking mode")
    
    # Sampling control
    parser.add_argument("--no-sampling", dest="do_sample", action="store_false", help="Disable sampling (use greedy)")
    parser.add_argument("--enable-sampling", dest="do_sample", action="store_true", help="Enable sampling")
    
    args = parser.parse_args()
    
    # Check server connectivity
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding at {args.url}")
            return 1
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server at {args.url}")
        return 1
    
    print(f"🚀 Enhanced Chat API CLI v2.0")
    print(f"🔗 Connected to: {args.url}")
    
    # Create and send request
    request = create_request(args)
    
    if args.details:
        print(f"\n📤 Request configuration:")
        for key, value in request.items():
            if key != "message":
                print(f"   {key}: {value}")
    
    print(f"\n💬 Sending message: {args.message}")
    print("⏳ Generating response...")
    
    response = send_request(request, args.url)
    if response:
        print_response(response, args.details)
    else:
        print("❌ Failed to get response")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
