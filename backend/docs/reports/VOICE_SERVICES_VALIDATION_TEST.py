#!/usr/bin/env python3
"""
Voice Services Validation Test

This script validates the BeautyAI voice services architecture and confirms:
1. Transcription factory creates the correct service
2. Voice config loader works properly
3. GPU-optimized transformer service is configured correctly
4. Server-side WebM/Opus decode environment is ready

Run with: python VOICE_SERVICES_VALIDATION_TEST.py
"""

import sys
import os
from pathlib import Path

# Add the backend src to path
backend_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

def test_voice_config_loader():
    """Test voice configuration loader."""
    print("🔧 Testing Voice Configuration Loader...")
    
    try:
        from beautyai_inference.config.voice_config_loader import get_voice_config
        
        vc = get_voice_config()
        
        # Test STT model config
        stt_config = vc.get_stt_model_config()
        print(f"  ✅ STT Model: {stt_config.model_id}")
        print(f"  ✅ Engine: {stt_config.engine_type}")
        print(f"  ✅ GPU Enabled: {stt_config.gpu_enabled}")
        
        # Test TTS model config
        tts_config = vc.get_tts_model_config()
        print(f"  ✅ TTS Model: {tts_config.model_id}")
        
        # Test audio config
        audio_config = vc.get_audio_config()
        print(f"  ✅ Audio Format: {audio_config.format} @ {audio_config.sample_rate}Hz")
        
        # Test voice settings
        ar_voices = vc.get_voice_settings("ar")
        en_voices = vc.get_voice_settings("en")
        print(f"  ✅ Arabic Voices: {len(ar_voices)} types")
        print(f"  ✅ English Voices: {len(en_voices)} types")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Voice Config Error: {e}")
        return False

def test_transcription_factory():
    """Test transcription service factory."""
    print("\n🏭 Testing Transcription Factory...")
    
    try:
        from beautyai_inference.services.voice.transcription.transcription_factory import create_transcription_service
        
        service = create_transcription_service()
        service_type = type(service).__name__
        print(f"  ✅ Created Service: {service_type}")
        
        # Test service methods exist
        assert hasattr(service, 'load_whisper_model'), "Missing load_whisper_model method"
        assert hasattr(service, 'transcribe_audio_bytes'), "Missing transcribe_audio_bytes method"
        assert hasattr(service, 'is_model_loaded'), "Missing is_model_loaded method"
        assert hasattr(service, 'get_model_info'), "Missing get_model_info method"
        print(f"  ✅ Service Interface: Complete")
        
        # Check if model is loaded (might not be loaded yet)
        is_loaded = service.is_model_loaded()
        print(f"  ℹ️ Model Loaded: {is_loaded}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Factory Error: {e}")
        return False

def test_environment_variables():
    """Test WebM/Opus decode environment variables."""
    print("\n🌍 Testing Environment Variables...")
    
    # Check key environment variables
    env_vars = {
        "VOICE_STREAMING_ENABLED": os.getenv("VOICE_STREAMING_ENABLED", "0"),
        "VOICE_STREAMING_PHASE4": os.getenv("VOICE_STREAMING_PHASE4", "0"),
        "VOICE_STREAMING_ALLOW_WEBM": os.getenv("VOICE_STREAMING_ALLOW_WEBM", "1"),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "auto"),
    }
    
    for var, value in env_vars.items():
        print(f"  ✅ {var}={value}")
    
    # Check streaming features
    streaming_enabled = env_vars["VOICE_STREAMING_ENABLED"] == "1"
    phase4_enabled = env_vars["VOICE_STREAMING_PHASE4"] == "1"
    webm_enabled = env_vars["VOICE_STREAMING_ALLOW_WEBM"] == "1"
    
    print(f"  ℹ️ Streaming Voice: {'Enabled' if streaming_enabled else 'Disabled'}")
    print(f"  ℹ️ Real Transcription: {'Enabled' if phase4_enabled else 'Mock Mode'}")
    print(f"  ℹ️ WebM/Opus Decode: {'Enabled' if webm_enabled else 'Disabled'}")
    
    return True

def test_ffmpeg_availability():
    """Test ffmpeg availability for WebM decode."""
    print("\n🎬 Testing FFmpeg Availability...")
    
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✅ FFmpeg Available: {version_line}")
            
            # Check for opus support
            if 'libopus' in result.stdout or 'opus' in result.stdout:
                print(f"  ✅ Opus Support: Available")
            else:
                print(f"  ⚠️ Opus Support: Unknown")
                
            return True
        else:
            print(f"  ❌ FFmpeg Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"  ❌ FFmpeg: Not found in PATH")
        return False
    except Exception as e:
        print(f"  ❌ FFmpeg Test Error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability for CUDA."""
    print("\n🚀 Testing GPU Availability...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"  ✅ CUDA Available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"  ✅ GPU Count: {device_count}")
            print(f"  ✅ Current Device: {current_device}")
            print(f"  ✅ Device Name: {device_name}")
            
            # Test memory
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
            
            print(f"  ℹ️ Memory Allocated: {memory_allocated:.2f} GB")
            print(f"  ℹ️ Memory Cached: {memory_cached:.2f} GB")
        
        return cuda_available
        
    except ImportError:
        print(f"  ❌ PyTorch: Not available")
        return False
    except Exception as e:
        print(f"  ❌ GPU Test Error: {e}")
        return False

def test_whisper_model_access():
    """Test if the whisper model can be accessed."""
    print("\n🎤 Testing Whisper Model Access...")
    
    try:
        # Test if we can import transformers
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        print(f"  ✅ Transformers: Available")
        
        # Test if we can create processor (without loading full model)
        model_id = "openai/whisper-large-v3-turbo"
        print(f"  ℹ️ Testing model access: {model_id}")
        
        # This will just check if the model exists on HuggingFace
        # without downloading the full model
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            print(f"  ✅ Model Config: Accessible")
            print(f"  ℹ️ Model Type: {config.model_type}")
            return True
        except Exception as e:
            print(f"  ⚠️ Model Access: {e}")
            return False
            
    except ImportError as e:
        print(f"  ❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Model Test Error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 BeautyAI Voice Services Validation Test")
    print("=" * 50)
    
    tests = [
        ("Voice Config Loader", test_voice_config_loader),
        ("Transcription Factory", test_transcription_factory),
        ("Environment Variables", test_environment_variables),
        ("FFmpeg Availability", test_ffmpeg_availability),
        ("GPU Availability", test_gpu_availability),
        ("Whisper Model Access", test_whisper_model_access),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  💥 {test_name} Failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Voice services are correctly configured.")
        return 0
    else:
        print("⚠️ Some tests failed. Check configuration and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())