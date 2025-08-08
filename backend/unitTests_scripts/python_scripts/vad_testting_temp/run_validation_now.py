#!/usr/bin/env python3

import subprocess
import sys
import os

def run_validation():
    """Run the comprehensive validation and show results"""
    print("🚀 Running comprehensive validation of backend chunk accumulation fix...")
    print("=" * 70)
    
    # Change to the correct directory
    os.chdir("/home/lumi/beautyai")
    
    try:
        # Run the validation script
        result = subprocess.run([
            sys.executable, "comprehensive_validation.py"
        ], capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Validation timed out")
        return False
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return False

if __name__ == "__main__":
    success = run_validation()
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}")
