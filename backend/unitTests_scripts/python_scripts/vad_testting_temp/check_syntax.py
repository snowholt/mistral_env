#!/usr/bin/env python3
"""
Quick syntax check for the updated backend file.
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check Python syntax of a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to parse the file
        ast.parse(content)
        print(f"✅ Syntax check passed: {file_path}")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")
        return False

def main():
    """Check syntax of key backend files"""
    print("🔍 Checking syntax of updated backend files...")
    
    files_to_check = [
        "/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py"
    ]
    
    all_good = True
    for file_path in files_to_check:
        if not check_syntax(file_path):
            all_good = False
    
    if all_good:
        print("\n✅ All syntax checks passed!")
    else:
        print("\n❌ Some syntax errors found!")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
