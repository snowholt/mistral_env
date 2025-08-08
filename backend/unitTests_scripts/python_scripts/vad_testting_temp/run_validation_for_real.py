#!/usr/bin/env python3

import subprocess
import sys
import os

print("🚀 ACTUALLY RUNNING THE VALIDATION NOW!")
print("="*50)

# Change to correct directory
os.chdir('/home/lumi/beautyai')

print("📁 Current directory:", os.getcwd())
print("📋 Running actual validation...")
print()

# Execute the validation script
try:
    result = subprocess.run([
        sys.executable, 'actual_validation_run.py'
    ], capture_output=False, text=True)
    
    print(f"\n✅ Validation completed with exit code: {result.returncode}")
    
except Exception as e:
    print(f"❌ Error running validation: {e}")

print("\n🔍 Let's also check what files we actually created:")
try:
    result = subprocess.run(['ls', '-la', '*.py'], capture_output=True, text=True, shell=True)
    print("Python files in directory:")
    print(result.stdout)
except Exception:
    pass

try:
    result = subprocess.run(['ls', '-la', '*.json'], capture_output=True, text=True, shell=True)
    print("JSON files in directory:")
    print(result.stdout)
except Exception:
    pass
