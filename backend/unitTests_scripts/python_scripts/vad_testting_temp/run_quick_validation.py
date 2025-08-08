#!/usr/bin/env python3

import subprocess
import sys

# Run the validation
print("🚀 Running backend chunk accumulation validation...")

try:
    result = subprocess.run([
        sys.executable, "/home/lumi/beautyai/validate_backend_fix.py"
    ], capture_output=True, text=True, cwd="/home/lumi/beautyai")
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"Exit code: {result.returncode}")

except Exception as e:
    print(f"Error running validation: {e}")
