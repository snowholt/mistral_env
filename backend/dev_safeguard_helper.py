"""
Developer Helper: Disable Safeguards for Testing

This module provides utility functions to bypass Kesay Clinics safeguards
during development and testing. 

⚠️ FOR TESTING ONLY - DO NOT USE IN PRODUCTION! ⚠️

Usage Examples:

1. Via API request:
   POST /api/chat
   {
       "message": "Tell me about the weather",
       "generation_config": {
           "disable_system_prompt_safeguards": true
       }
   }

2. Via environment variable:
   export DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1
   sudo systemctl restart beautyai-api.service

3. Via Python code:
   from dev_safeguard_helper import with_safeguards_disabled
   
   @with_safeguards_disabled
   def test_unrestricted_chat():
       # Your test code here
       pass

Created by: Lumina Ashley 💕
"""

import os
import functools
from typing import Dict, Any, Callable


def set_safeguards_disabled(disabled: bool = True) -> None:
    """
    Set the safeguard bypass flag in environment.
    
    Args:
        disabled: True to disable safeguards, False to enable them
    """
    if disabled:
        os.environ['DISABLE_SYSTEM_PROMPT_SAFEGUARDS'] = '1'
        print("🔓 Safeguards DISABLED (Developer Mode)")
    else:
        os.environ.pop('DISABLE_SYSTEM_PROMPT_SAFEGUARDS', None)
        print("🔒 Safeguards ENABLED (Production Mode)")


def are_safeguards_disabled() -> bool:
    """
    Check if safeguards are currently disabled.
    
    Returns:
        True if safeguards are disabled, False otherwise
    """
    return os.getenv('DISABLE_SYSTEM_PROMPT_SAFEGUARDS', '').lower() in ('1', 'true', 'yes')


def with_safeguards_disabled(func: Callable) -> Callable:
    """
    Decorator to temporarily disable safeguards for a function.
    
    Usage:
        @with_safeguards_disabled
        def my_test_function():
            # Code here runs without safeguards
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Save original state
        original_state = are_safeguards_disabled()
        
        try:
            # Disable safeguards
            set_safeguards_disabled(True)
            # Run function
            return func(*args, **kwargs)
        finally:
            # Restore original state
            set_safeguards_disabled(original_state)
    
    return wrapper


def add_safeguard_bypass_to_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add safeguard bypass flag to generation config.
    
    Args:
        config: Generation configuration dictionary
        
    Returns:
        Updated configuration with bypass flag
    """
    config = config.copy()
    config['disable_system_prompt_safeguards'] = True
    return config


# Quick test functions
def test_safeguard_status():
    """Quick test to check safeguard status."""
    print("\n🧪 Testing Safeguard Status")
    print("="*50)
    
    if are_safeguards_disabled():
        print("Status: 🔓 DISABLED (Developer Mode)")
        print("Model behavior:")
        print("  ✅ Can discuss any topic")
        print("  ✅ Responds in detected language")
        print("  ✅ No Kesay Clinics restrictions")
    else:
        print("Status: 🔒 ENABLED (Production Mode)")
        print("Model behavior:")
        print("  ⚠️  Only Kesay Clinics topics")
        print("  ⚠️  Arabic-only responses")
        print("  ⚠️  Doctor info required")
    
    print("="*50)


def example_chat_request_with_bypass() -> Dict[str, Any]:
    """
    Example chat request payload with safeguard bypass.
    
    Returns:
        Example request payload
    """
    return {
        "model_name": "qwen3-unsloth-q4ks",
        "message": "What's the weather like today?",
        "generation_config": {
            "max_tokens": 200,
            "temperature": 0.7,
            "disable_system_prompt_safeguards": True  # 🔓 Bypass flag
        }
    }


def example_voice_test_with_bypass():
    """
    Example of how to set up voice testing without safeguards.
    """
    print("\n🎤 Voice Testing Setup (No Safeguards)")
    print("="*50)
    print("1. Set environment variable:")
    print("   export DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1")
    print("")
    print("2. Restart API service:")
    print("   sudo systemctl restart beautyai-api.service")
    print("")
    print("3. Run voice test:")
    print("   python tests/streaming/ws_replay_pcm.py --file voice_tests/input_test_questions/pcm/q1.pcm")
    print("")
    print("4. After testing, re-enable safeguards:")
    print("   unset DISABLE_SYSTEM_PROMPT_SAFEGUARDS")
    print("   sudo systemctl restart beautyai-api.service")
    print("="*50)


if __name__ == "__main__":
    print("🔐 Safeguard Helper Utilities")
    print("")
    test_safeguard_status()
    print("")
    print("📖 Example Usage:")
    print("")
    print("# Disable safeguards programmatically")
    print("from dev_safeguard_helper import set_safeguards_disabled")
    print("set_safeguards_disabled(True)")
    print("")
    print("# Add bypass to API request")
    print("from dev_safeguard_helper import example_chat_request_with_bypass")
    print("request = example_chat_request_with_bypass()")
    print("")
    print("# Use decorator for tests")
    print("from dev_safeguard_helper import with_safeguards_disabled")
    print("")
    print("@with_safeguards_disabled")
    print("def my_test():")
    print("    # Test code here")
    print("    pass")
