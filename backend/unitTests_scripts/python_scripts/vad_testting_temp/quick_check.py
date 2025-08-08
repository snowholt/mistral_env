#!/usr/bin/env python3

import ast
from pathlib import Path

def quick_syntax_check():
    """Quick syntax check of the backend file"""
    print("🔍 Quick Backend File Syntax Check")
    print("=" * 40)
    
    file_path = "/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py"
    
    if not Path(file_path).exists():
        print("❌ Backend file not found")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check syntax
        ast.parse(content)
        print("✅ Python syntax: VALID")
        
        # Check key components
        components = {
            "chunk_buffer": "chunk_buffer" in content,
            "_process_buffered_chunks": "def _process_buffered_chunks" in content,
            "_setup_vad_callbacks": "def _setup_vad_callbacks" in content,
            "processing_turn": "processing_turn" in content,
            "WebM concatenation": "b''.join(connection[\"chunk_buffer\"])" in content
        }
        
        print("\n📋 Component Check:")
        all_good = True
        for component, present in components.items():
            status = "✅" if present else "❌"
            print(f"  {status} {component}")
            if not present:
                all_good = False
        
        print(f"\n🎯 Overall: {'✅ PASS' if all_good else '❌ FAIL'}")
        return all_good
        
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_syntax_check()
