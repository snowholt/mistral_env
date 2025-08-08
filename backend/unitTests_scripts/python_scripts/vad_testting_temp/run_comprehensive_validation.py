#!/usr/bin/env python3

import subprocess
import sys

def run_validation():
    """Run the comprehensive validation script"""
    print("🚀 Running comprehensive backend validation...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "/home/lumi/beautyai/comprehensive_validation.py"
        ], cwd="/home/lumi/beautyai", capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = run_validation()
    print(f"\n{'✅ Validation completed successfully!' if success else '❌ Validation encountered issues.'}")
