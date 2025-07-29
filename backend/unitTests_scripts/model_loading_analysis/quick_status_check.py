#!/usr/bin/env python3
"""
Quick Model Status Checker

This script provides a quick check of the current model loading status
across all BeautyAI services. It helps verify the issue before running
comprehensive tests.

Author: BeautyAI Framework
Date: 2025-07-27
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path
import time

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def check_model_status(base_url: str = "http://localhost:8000"):
    """Check current model loading status."""
    
    print("🔍 BeautyAI Model Loading Status Check")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # 1. Check API health
        print("1️⃣ Checking API Health...")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"   ✅ API is running: {health_data.get('status', 'unknown')}")
                else:
                    print(f"   ❌ API health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"   ❌ Cannot connect to API: {e}")
            print(f"   💡 Make sure BeautyAI API is running at {base_url}")
            return
        
        # 2. Check model status
        print("\n2️⃣ Checking Model Loading Status...")
        try:
            async with session.get(f"{base_url}/models/status") as response:
                if response.status == 200:
                    model_data = await response.json()
                    total_loaded = model_data.get("total_loaded", 0)
                    models = model_data.get("models", [])
                    
                    print(f"   📊 Total Models Loaded: {total_loaded}")
                    
                    if models:
                        print("   📋 Loaded Models:")
                        for model in models:
                            print(f"      • {model}")
                    else:
                        print("   ⚠️ NO MODELS ARE CURRENTLY LOADED!")
                        print("   🎯 This confirms the issue - models are not pre-loaded")
                else:
                    print(f"   ❌ Failed to get model status: {response.status}")
        except Exception as e:
            print(f"   ❌ Error checking model status: {e}")
        
        # 3. Check available models
        print("\n3️⃣ Checking Available Models...")
        try:
            async with session.get(f"{base_url}/models") as response:
                if response.status == 200:
                    available_models = await response.json()
                    if "models" in available_models:
                        total_available = len(available_models["models"])
                        print(f"   📚 Total Available Models: {total_available}")
                        
                        # Show key models
                        key_models = ["qwen3-unsloth-q4ks", "whisper-large-v3-turbo-arabic", "coqui-tts-arabic"]
                        print("   🔑 Key Models for Voice Services:")
                        for model_name in key_models:
                            found = any(m.get("name") == model_name for m in available_models["models"])
                            status = "✅ Available" if found else "❌ Missing"
                            print(f"      • {model_name}: {status}")
                    else:
                        print("   ⚠️ No model registry data found")
                else:
                    print(f"   ❌ Failed to get available models: {response.status}")
        except Exception as e:
            print(f"   ❌ Error checking available models: {e}")
        
        # 4. Check WebSocket service status
        print("\n4️⃣ Checking WebSocket Service Status...")
        
        # Simple Voice WebSocket
        try:
            async with session.get(f"{base_url}/ws/simple-voice-chat/status") as response:
                if response.status == 200:
                    simple_status = await response.json()
                    connections = simple_status.get("active_connections", 0)
                    print(f"   🎤 Simple Voice WebSocket: ✅ Available ({connections} active connections)")
                else:
                    print(f"   🎤 Simple Voice WebSocket: ❌ Unavailable ({response.status})")
        except Exception as e:
            print(f"   🎤 Simple Voice WebSocket: ❌ Error ({e})")
        
        # Advanced Voice WebSocket
        try:
            async with session.get(f"{base_url}/ws/voice-conversation/status") as response:
                if response.status == 200:
                    advanced_status = await response.json()
                    connections = advanced_status.get("active_connections", 0)
                    print(f"   🎯 Advanced Voice WebSocket: ✅ Available ({connections} active connections)")
                else:
                    print(f"   🎯 Advanced Voice WebSocket: ❌ Unavailable ({response.status})")
        except Exception as e:
            print(f"   🎯 Advanced Voice WebSocket: ❌ Error ({e})")
        
        # 5. Test a quick chat request to see loading behavior
        print("\n5️⃣ Testing Chat API Response Time...")
        try:
            test_request = {
                "model_name": "qwen3-unsloth-q4ks",
                "message": "مرحبا",
                "max_new_tokens": 10
            }
            
            start_time = time.time()
            async with session.post(
                f"{base_url}/inference/chat",
                json=test_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    print(f"   ⚡ Chat API Response Time: {response_time:.2f}s")
                    
                    if response_time > 20:
                        print("   🐌 SLOW RESPONSE - Model likely loaded on-demand")
                    elif response_time < 5:
                        print("   🚀 FAST RESPONSE - Model likely pre-loaded")
                    else:
                        print("   🤔 MODERATE RESPONSE - Unclear if model was pre-loaded")
                    
                    print(f"   💬 Response: {result.get('response', '')[:50]}...")
                else:
                    print(f"   ❌ Chat API test failed: {response.status}")
        except Exception as e:
            print(f"   ❌ Chat API test error: {e}")
    
    # 6. Summary and recommendations
    print("\n" + "=" * 50)
    print("📋 SUMMARY & NEXT STEPS")
    print("=" * 50)
    print("🔍 To run comprehensive analysis:")
    print("   python test_model_persistence.py")
    print("\n📊 To monitor model loading in real-time:")
    print("   python monitor_model_loading.py")
    print("\n🧪 To test WebSocket specifically:")
    print("   python monitor_model_loading.py --test websocket --duration 60")


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick BeautyAI Model Status Check")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    await check_model_status(args.url)

if __name__ == "__main__":
    asyncio.run(main())
