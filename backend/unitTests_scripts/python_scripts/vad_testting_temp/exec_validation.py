#!/usr/bin/env python3

import subprocess
import sys
import os

print("🧪 RUNNING BACKEND CHUNK ACCUMULATION VALIDATION")
print("=" * 60)

# Change to correct directory
os.chdir("/home/lumi/beautyai")

# Execute the comprehensive validation directly
try:
    exec(open("comprehensive_validation.py").read())
except Exception as e:
    print(f"❌ Error executing validation: {e}")
    sys.exit(1)
