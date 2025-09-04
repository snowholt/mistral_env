#!/usr/bin/env python3
"""Basic validation test for duplex streaming components - runs without pytest."""

import sys
import os
import asyncio
import numpy as np
from typing import Dict, Any

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

try:
    # Test imports
    from beautyai_inference.services.voice.echo_suppression import EchoSuppressor, EchoState
    from beautyai_inference.services.voice.utils.echo_detector import EchoDetector, create_echo_detector
    from beautyai_inference.services.voice.streaming.metrics import SessionMetrics
    from beautyai_inference.api.adapters.config_adapter import ConfigAdapter
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_echo_detector_creation():
    """Test echo detector creation and basic functionality."""
    print("Testing echo detector creation...")
    
    try:
        # Test simple detector
        simple_detector = create_echo_detector("simple")
        assert simple_detector is not None
        print("✅ Simple detector created")
        
        # Test advanced detector  
        advanced_detector = create_echo_detector("advanced")
        assert advanced_detector is not None
        print("✅ Advanced detector created")
        
        # Test processing with dummy data
        mic_data = np.random.random(1024).astype(np.float32)
        tts_data = np.random.random(1024).astype(np.float32)
        
        result = simple_detector.process_audio(mic_data, tts_data)
        assert "echo_probability" in result
        assert "correlation_score" in result
        assert "spectral_similarity" in result
        assert "confidence" in result
        print("✅ Echo detector processes audio correctly")
        
    except Exception as e:
        print(f"❌ Echo detector test failed: {e}")
        return False
    
    return True

def test_echo_suppressor_creation():
    """Test echo suppressor creation and state management."""
    print("Testing echo suppressor creation...")
    
    try:
        # Create with default config
        suppressor = EchoSuppressor()
        assert suppressor.get_state() == EchoState.IDLE
        print("✅ Echo suppressor created in IDLE state")
        
        # Test state transitions
        suppressor.start_tts_playback()
        assert suppressor.get_state() == EchoState.TTS_PLAYING
        print("✅ TTS playback state transition works")
        
        suppressor.stop_tts_playbook()
        assert suppressor.get_state() == EchoState.IDLE
        print("✅ TTS stop state transition works")
        
        # Test metrics
        metrics = suppressor.get_metrics()
        assert isinstance(metrics.session_start_time, float)
        print("✅ Metrics retrieval works")
        
        # Test statistics
        stats = suppressor.get_echo_statistics()
        assert "echo_suppression" in stats
        assert "echo_detection" in stats
        assert "current_metrics" in stats
        print("✅ Echo statistics retrieval works")
        
    except Exception as e:
        print(f"❌ Echo suppressor test failed: {e}")
        return False
        
    return True

def test_metrics_creation():
    """Test session metrics creation."""
    print("Testing session metrics...")
    
    try:
        metrics = SessionMetrics()
        assert hasattr(metrics, 'duplex')
        assert hasattr(metrics.duplex, 'echo_correlations')
        print("✅ Session metrics created with duplex support")
        
        # Test metrics update
        metrics.duplex.echo_correlations.append(0.5)
        metrics.duplex.barge_in_events.append({'timestamp': 123456, 'type': 'start'})
        
        snapshot = metrics.get_snapshot()
        assert 'duplex' in snapshot
        print("✅ Metrics snapshot includes duplex data")
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False
        
    return True

def test_config_validation():
    """Test config adapter validation."""
    print("Testing config validation...")
    
    try:
        adapter = ConfigAdapter()
        
        # Test duplex config validation
        config = {
            "duplex_streaming": {
                "enabled": True,
                "echo_suppression": {
                    "enabled": True,
                    "detector_type": "simple",
                    "detection_threshold": 0.6
                }
            }
        }
        
        # This should not raise an exception
        validated = adapter._validate_inference_config(config)
        print("✅ Config validation works")
        
    except Exception as e:
        print(f"❌ Config validation test failed: {e}")
        return False
        
    return True

def main():
    """Run all basic validation tests."""
    print("=== Duplex Streaming Basic Validation ===")
    
    tests = [
        test_echo_detector_creation,
        test_echo_suppressor_creation, 
        test_metrics_creation,
        test_config_validation
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
        print("\n🎉 All tests passed! Duplex streaming implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())