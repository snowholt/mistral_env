#!/usr/bin/env python3

import ast
import subprocess
import sys
from pathlib import Path

def check_syntax():
    """Check the syntax of our updated backend file"""
    file_path = "/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py"
    
    print(f"🔍 Checking syntax of: {file_path}")
    
    try:
        # Check Python syntax
        result = subprocess.run([
            sys.executable, "-m", "py_compile", file_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Python syntax check: PASSED")
            return True
        else:
            print("❌ Python syntax check: FAILED")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error checking syntax: {e}")
        return False

def check_imports():
    """Check if the file can be imported (basic validation)"""
    print("🔍 Checking imports and basic structure...")
    
    try:
        # Try to parse with AST
        with open("/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py", 'r') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ AST parsing: PASSED")
        
        # Check for key methods
        key_methods = [
            "process_realtime_audio_chunk",
            "_process_buffered_chunks", 
            "_setup_vad_callbacks"
        ]
        
        for method in key_methods:
            if f"def {method}" in content:
                print(f"✅ Method {method}: FOUND")
            else:
                print(f"❌ Method {method}: MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import/structure check failed: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🧪 Backend File Validation")
    print("=" * 40)
    
    syntax_ok = check_syntax()
    structure_ok = check_imports()
    
    print("\n" + "=" * 40)
    print("📋 VALIDATION RESULTS")
    print("=" * 40)
    
    if syntax_ok and structure_ok:
        print("✅ Backend file validation: PASSED")
        print("   - No syntax errors")
        print("   - All required methods present")
        print("   - Ready for testing")
        return True
    else:
        print("❌ Backend file validation: FAILED")
        if not syntax_ok:
            print("   - Syntax errors found")
        if not structure_ok:
            print("   - Missing required methods")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
