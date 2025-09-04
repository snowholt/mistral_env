#!/usr/bin/env python3
"""Direct import test for duplex streaming components."""

import sys
import os

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

def test_echo_detector_direct():
    """Test echo detector module directly."""
    print("Testing echo detector direct import...")
    
    try:
        import beautyai_inference.services.voice.utils.echo_detector as echo_detector_module
        print("✅ Echo detector module imported directly")
        
        # Check classes exist
        assert hasattr(echo_detector_module, 'EchoDetector')
        assert hasattr(echo_detector_module, 'SimpleEchoDetector') 
        assert hasattr(echo_detector_module, 'AdvancedEchoDetector')
        assert hasattr(echo_detector_module, 'create_echo_detector')
        print("✅ All expected classes found")
        
        # Test factory function
        simple = echo_detector_module.create_echo_detector("simple")
        advanced = echo_detector_module.create_echo_detector("advanced")
        print("✅ Factory function works")
        
        return True
        
    except Exception as e:
        print(f"❌ Echo detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_direct():
    """Test metrics module directly."""
    print("Testing metrics direct import...")
    
    try:
        import beautyai_inference.services.voice.streaming.metrics as metrics_module
        print("✅ Metrics module imported directly")
        
        # Check classes exist
        assert hasattr(metrics_module, 'SessionMetrics')
        assert hasattr(metrics_module, 'DuplexMetrics')
        print("✅ All expected classes found")
        
        # Test creation
        session_metrics = metrics_module.SessionMetrics()
        assert hasattr(session_metrics, 'duplex')
        print("✅ SessionMetrics with duplex support created")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_echo_suppression_direct():
    """Test echo suppression module directly.""" 
    print("Testing echo suppression direct import...")
    
    try:
        import beautyai_inference.services.voice.echo_suppression as echo_suppression_module
        print("✅ Echo suppression module imported directly")
        
        # Check classes/enums exist
        assert hasattr(echo_suppression_module, 'EchoState')
        assert hasattr(echo_suppression_module, 'EchoSuppressor')
        print("✅ All expected classes found")
        
        # Test enum
        states = [
            echo_suppression_module.EchoState.IDLE,
            echo_suppression_module.EchoState.TTS_PLAYING,
            echo_suppression_module.EchoState.USER_SPEAKING,
            echo_suppression_module.EchoState.BARGE_IN
        ]
        print(f"✅ Echo states: {[s.value for s in states]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Echo suppression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_adapter_direct():
    """Test config adapter module directly."""
    print("Testing config adapter direct import...")
    
    try:
        # Import just the specific module file
        sys.path.append('/home/lumi/beautyai/backend/src/beautyai_inference/api/adapters')
        import config_adapter as config_module
        print("✅ Config adapter module imported directly")
        
        # Check class exists
        assert hasattr(config_module, 'ConfigAdapter')
        print("✅ ConfigAdapter class found")
        
        return True
        
    except Exception as e:
        print(f"❌ Config adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all direct import tests."""
    print("=== Duplex Streaming Direct Import Validation ===")
    
    tests = [
        test_echo_detector_direct,
        test_metrics_direct,
        test_echo_suppression_direct,
        test_config_adapter_direct
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
        print("\n🎉 All direct import tests passed! New modules are correctly structured.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review module structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main())