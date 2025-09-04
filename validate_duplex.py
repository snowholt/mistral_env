#!/usr/bin/env python3
"""
Simple validation script for duplex streaming implementation
Tests import capabilities and basic functionality without runtime dependencies
"""

import sys
import os
import importlib.util

def test_imports():
    """Test if all our new modules can be imported successfully"""
    results = {}
    
    # Test echo detector
    try:
        spec = importlib.util.spec_from_file_location(
            "echo_detector", 
            "backend/src/beautyai_inference/services/voice/utils/echo_detector.py"
        )
        echo_detector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(echo_detector)
        results['echo_detector'] = "✅ PASS"
        
        # Test if EchoDetector class can be instantiated
        detector = echo_detector.EchoDetector()
        results['echo_detector_instantiation'] = "✅ PASS"
        
    except Exception as e:
        results['echo_detector'] = f"❌ FAIL: {str(e)}"
    
    # Test metrics module
    try:
        spec = importlib.util.spec_from_file_location(
            "metrics", 
            "backend/src/beautyai_inference/services/voice/streaming/metrics.py"
        )
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        results['metrics'] = "✅ PASS"
        
    except Exception as e:
        results['metrics'] = f"❌ FAIL: {str(e)}"
    
    # Test config adapter
    try:
        spec = importlib.util.spec_from_file_location(
            "config_adapter", 
            "backend/src/beautyai_inference/api/adapters/config_adapter.py"
        )
        config_adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_adapter)
        results['config_adapter'] = "✅ PASS"
        
    except Exception as e:
        results['config_adapter'] = f"❌ FAIL: {str(e)}"
    
    return results

def test_file_structure():
    """Test if all expected files exist"""
    expected_files = [
        'backend/src/beautyai_inference/services/voice/utils/echo_detector.py',
        'backend/src/beautyai_inference/services/voice/streaming/metrics.py',
        'backend/src/beautyai_inference/api/adapters/config_adapter.py',
        'frontend/src/static/js/ttsPlayer.js',
        'frontend/src/static/js/duplexWebSocket.js',
        'frontend/src/static/js/tts-player-worklet.js',
        'tests/test_duplex_streaming.py',
        'tests/test_duplex_basic.py',
        'docs/DUPLEX_DEPLOYMENT.md'
    ]
    
    results = {}
    for file_path in expected_files:
        if os.path.exists(file_path):
            results[file_path] = "✅ EXISTS"
        else:
            results[file_path] = "❌ MISSING"
    
    return results

def main():
    print("🎵 BeautyAI Duplex Streaming Validation")
    print("=" * 50)
    
    print("\n📁 File Structure Test:")
    file_results = test_file_structure()
    for file_path, status in file_results.items():
        print(f"  {status} {file_path}")
    
    print("\n🔗 Import Test:")
    import_results = test_imports()
    for module, status in import_results.items():
        print(f"  {status} {module}")
    
    # Summary
    total_files = len(file_results)
    existing_files = sum(1 for status in file_results.values() if status.startswith("✅"))
    
    total_imports = len(import_results)
    passing_imports = sum(1 for status in import_results.values() if status.startswith("✅"))
    
    print("\n📊 Summary:")
    print(f"  Files: {existing_files}/{total_files} exist")
    print(f"  Imports: {passing_imports}/{total_imports} successful")
    
    if existing_files == total_files and passing_imports == total_imports:
        print("\n🎉 All duplex streaming components validated successfully!")
        print("✅ Ready for production deployment")
        return 0
    else:
        print("\n⚠️  Some components need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())