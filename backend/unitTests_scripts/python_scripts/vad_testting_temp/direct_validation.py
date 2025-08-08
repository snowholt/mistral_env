#!/usr/bin/env python3

import os
import sys

# Change to the beautyai directory
os.chdir('/home/lumi/beautyai')

# Add to Python path
sys.path.insert(0, '/home/lumi/beautyai')

print("🚀 Executing comprehensive validation...")
print("Current directory:", os.getcwd())
print("Python path:", sys.path[0])

try:
    # Import and run the validation
    import comprehensive_validation
    print("✅ Validation script imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    
    # Try direct execution
    print("\n🔄 Trying direct execution...")
    try:
        with open('comprehensive_validation.py', 'r') as f:
            code = f.read()
        exec(code)
    except Exception as e2:
        print(f"❌ Direct execution failed: {e2}")
