#!/usr/bin/env python3
"""Minimal validation test for duplex streaming components - no external dependencies."""

import sys
import os

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

def test_basic_imports():
    """Test that all our new modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from beautyai_inference.services.voice.echo_suppression import EchoSuppressor, EchoState
        print("✅ EchoSuppressor imported")
        
        from beautyai_inference.services.voice.utils.echo_detector import EchoDetector, create_echo_detector
        print("✅ EchoDetector imported")
        
        from beautyai_inference.services.voice.streaming.metrics import SessionMetrics  
        print("✅ SessionMetrics imported")
        
        from beautyai_inference.api.adapters.config_adapter import ConfigAdapter
        print("✅ ConfigAdapter imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_echo_state_enum():
    """Test echo state enumeration."""
    print("Testing echo state enum...")
    
    try:
        from beautyai_inference.services.voice.echo_suppression import EchoState
        
        # Test all expected states exist
        states = [EchoState.IDLE, EchoState.TTS_PLAYING, EchoState.USER_SPEAKING, EchoState.BARGE_IN]
        print(f"✅ All echo states available: {[s.value for s in states]}")
        return True
        
    except Exception as e:
        print(f"❌ Echo state test failed: {e}")
        return False

def test_echo_detector_factory():
    """Test echo detector factory function."""
    print("Testing echo detector factory...")
    
    try:
        from beautyai_inference.services.voice.utils.echo_detector import create_echo_detector
        
        # Test valid detector types
        simple_detector = create_echo_detector("simple")
        print("✅ Simple detector created")
        
        advanced_detector = create_echo_detector("advanced")  
        print("✅ Advanced detector created")
        
        # Test invalid detector type
        try:
            invalid_detector = create_echo_detector("nonexistent")
            print("❌ Should have failed for invalid detector type")
            return False
        except ValueError:
            print("✅ Correctly rejected invalid detector type")
        
        return True
        
    except Exception as e:
        print(f"❌ Echo detector factory test failed: {e}")
        return False

def test_echo_suppressor_creation():
    """Test basic echo suppressor creation."""
    print("Testing echo suppressor creation...")
    
    try:
        from beautyai_inference.services.voice.echo_suppression import EchoSuppressor, EchoState
        
        # Create with default config
        suppressor = EchoSuppressor()
        print("✅ EchoSuppressor created")
        
        # Check initial state
        state = suppressor.get_state()
        if state == EchoState.IDLE:
            print("✅ Initial state is IDLE")
        else:
            print(f"❌ Initial state should be IDLE, got {state}")
            return False
            
        # Test state transitions
        suppressor.start_tts_playback()
        if suppressor.get_state() == EchoState.TTS_PLAYING:
            print("✅ TTS start transition works")
        else:
            print(f"❌ Expected TTS_PLAYING, got {suppressor.get_state()}")
            return False
            
        suppressor.stop_tts_playbook()
        if suppressor.get_state() == EchoState.IDLE:
            print("✅ TTS stop transition works")
        else:
            print(f"❌ Expected IDLE after stop, got {suppressor.get_state()}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Echo suppressor creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic validation tests."""
    print("=== Duplex Streaming Minimal Validation ===")
    
    tests = [
        test_basic_imports,
        test_echo_state_enum,
        test_echo_detector_factory,
        test_echo_suppressor_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with exception: {e}")
    
    print(f"\n=== Results ===")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {len(tests)}")
    
    if failed == 0:
        print("\n🎉 All basic tests passed! Core components are working.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())