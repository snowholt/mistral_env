#!/usr/bin/env python3
"""
Basic Coqui TTS test to diagnose model loading issues.
"""

import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_coqui_basic():
    """Test basic Coqui TTS functionality."""
    try:
        print("🔍 Testing Coqui TTS Basic Functionality")
        print("=" * 50)
        
        # Test import
        print("📦 Testing Coqui TTS import...")
        from TTS.api import TTS
        print("✅ TTS import successful")
        
        # List available models
        print("\n📋 Listing available models...")
        available_models = TTS.list_models()
        print(f"Total models available: {len(available_models)}")
        
        # Filter Arabic models
        arabic_models = [model for model in available_models if 'ar/' in model or 'arabic' in model.lower()]
        print(f"\n🇸🇦 Arabic models found: {len(arabic_models)}")
        for model in arabic_models:
            print(f"  - {model}")
        
        # Filter multilingual models
        multilingual_models = [model for model in available_models if 'multilingual' in model.lower()]
        print(f"\n🌍 Multilingual models found: {len(multilingual_models)}")
        for model in multilingual_models[:5]:  # Show first 5
            print(f"  - {model}")
        
        # Test loading a simple Arabic model
        print("\n🚀 Testing Arabic model loading...")
        if arabic_models:
            test_model = arabic_models[0]
            print(f"Loading model: {test_model}")
            
            try:
                tts = TTS(model_name=test_model, gpu=False)  # Use CPU first
                print(f"✅ Successfully loaded: {test_model}")
                
                # Test speakers if available
                if hasattr(tts, 'speakers') and tts.speakers:
                    print(f"Available speakers: {len(tts.speakers)}")
                    print(f"Speaker list: {tts.speakers[:5]}")  # First 5 speakers
                else:
                    print("No speakers information available")
                
                # Test basic TTS
                print("\n🎤 Testing text synthesis...")
                test_text = "مرحبا بك في نظام الذكاء الاصطناعي"
                output_path = "/home/lumi/beautyai/test_basic_arabic.wav"
                
                tts.tts_to_file(text=test_text, file_path=output_path)
                print(f"✅ TTS synthesis successful: {output_path}")
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to load model {test_model}: {e}")
                
                # Try multilingual as fallback
                if multilingual_models:
                    print(f"\n🔄 Trying multilingual model as fallback...")
                    fallback_model = multilingual_models[0]
                    try:
                        tts = TTS(model_name=fallback_model, gpu=False)
                        print(f"✅ Successfully loaded fallback: {fallback_model}")
                        
                        # Test with Arabic text
                        test_text = "مرحبا بك في نظام الذكاء الاصطناعي"
                        output_path = "/home/lumi/beautyai/test_multilingual_arabic.wav"
                        
                        # For multilingual models, we might need to specify language
                        if 'xtts' in fallback_model:
                            tts.tts_to_file(text=test_text, file_path=output_path, language="ar")
                        else:
                            tts.tts_to_file(text=test_text, file_path=output_path)
                        
                        print(f"✅ Multilingual TTS synthesis successful: {output_path}")
                        return True
                        
                    except Exception as e2:
                        print(f"❌ Fallback also failed: {e2}")
                        return False
        else:
            print("❌ No Arabic models found")
            return False
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def test_installation():
    """Test Coqui TTS installation and dependencies."""
    print("🔧 Testing Coqui TTS Installation")
    print("=" * 50)
    
    try:
        import TTS
        print(f"✅ TTS version: {TTS.__version__}")
    except Exception as e:
        print(f"❌ TTS import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
    
    return True

if __name__ == "__main__":
    print("🧪 Coqui TTS Diagnostic Test")
    print("=" * 60)
    
    # Test installation first
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n")
    
    # Test basic functionality
    if test_coqui_basic():
        print("\n✅ All tests passed! Coqui TTS is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for issues.")
        sys.exit(1)
