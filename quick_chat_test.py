#!/usr/bin/env python3
"""
Quick speed test for chat interface with optimized model.
"""

import subprocess
import sys
import os

def test_optimized_chat():
    """Test the chat interface with the optimized model."""
    
    print("🚀 Testing optimized chat interface...")
    print("Model: bee1reason-arabic-qwen-14b-i1q4ks-gguf")
    print("Expected speed: 70-100+ tokens/second")
    print("-" * 60)
    
    # Run the CLI with the optimized model
    cmd = [
        sys.executable, 
        "-m", "beautyai_inference.cli.unified_cli",
        "chat",
        "--model", "bee1reason-arabic-qwen-14b-i1q4ks-gguf",
        "--max-tokens", "32",  # Very short for maximum speed
        "--temperature", "0.01",
        "--top-p", "0.5"
    ]
    
    print("Command:", " ".join(cmd))
    print("\nStarting optimized chat interface...")
    print("Try asking: 'What is AI?' for a quick speed test")
    print("Type 'exit' to quit")
    print("=" * 60)
    
    try:
        # Change to the project directory
        os.chdir("/home/lumi/beautyai")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✅ Chat session ended.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_optimized_chat()
