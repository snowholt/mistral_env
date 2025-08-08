#!/usr/bin/env python3

import os
import sys

# Change to correct directory
os.chdir('/home/lumi/beautyai')

# Execute the manual validation
print("🚀 EXECUTING MANUAL VALIDATION...")
print()

try:
    exec(open('manual_validation.py').read())
except Exception as e:
    print(f"❌ Error during validation: {e}")
    import traceback
    traceback.print_exc()
