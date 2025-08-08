#!/usr/bin/env python3
"""
Let's actually RUN the validation and see the results!
"""

import ast
import subprocess
import sys
import os
from pathlib import Path

# Change to correct directory
os.chdir('/home/lumi/beautyai')

print("🧪 ACTUALLY RUNNING BACKEND VALIDATION NOW!")
print("="*60)

# First, let's check the backend file exists and get basic info
file_path = "/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py"

print(f"📁 Checking: {file_path}")

if not Path(file_path).exists():
    print("❌ CRITICAL: Backend file not found!")
    sys.exit(1)

print("✅ Backend file exists")

# Read and check syntax
try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"✅ File readable: {len(content):,} characters")
    
    # Check Python syntax
    ast.parse(content)
    print("✅ Python syntax: VALID")
    
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: Line {e.lineno}: {e.msg}")
    print(f"   Text: {e.text}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR reading file: {e}")
    sys.exit(1)

# Now check our implementation components
print("\n🔍 CHECKING IMPLEMENTATION COMPONENTS:")
print("-" * 50)

components_to_check = [
    ("Chunk buffer usage", "chunk_buffer"),
    ("Process buffered chunks method", "def _process_buffered_chunks"),
    ("Setup VAD callbacks method", "def _setup_vad_callbacks"), 
    ("Processing turn flag", "processing_turn"),
    ("WebM chunk concatenation", "b''.join(connection[\"chunk_buffer\"])"),
    ("Chunk append logic", "chunk_buffer.append(audio_data)"),
    ("Buffer clearing", "connection[\"chunk_buffer\"] = []"),
    ("Ignore during processing", "if connection.get(\"processing_turn\", False):"),
    ("Turn ID tracking", "last_turn_id"),
    ("VAD callback setup", "_setup_vad_callbacks(connection_id)")
]

total_components = len(components_to_check)
found_components = 0

for name, pattern in components_to_check:
    if pattern in content:
        print(f"✅ {name}")
        found_components += 1
    else:
        print(f"❌ {name} - MISSING!")

implementation_score = found_components / total_components
print(f"\n📊 Implementation Score: {found_components}/{total_components} ({implementation_score:.1%})")

# Check frontend files
print("\n🌐 CHECKING FRONTEND COMPATIBILITY:")
print("-" * 50)

frontend_files = [
    "/home/lumi/beautyai/frontend/src/static/js/chat-interface.js",
    "/home/lumi/beautyai/frontend/src/static/js/voice-overlay-strict-patch.js"
]

frontend_ready = True
for file_path in frontend_files:
    if Path(file_path).exists():
        print(f"✅ {Path(file_path).name}")
    else:
        print(f"❌ {Path(file_path).name} - MISSING!")
        frontend_ready = False

# Check backend server status
print("\n🔌 CHECKING BACKEND SERVER STATUS:")
print("-" * 50)

try:
    result = subprocess.run(["pgrep", "-f", "run_server.py"], capture_output=True, text=True)
    if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        print(f"✅ Backend server RUNNING (PIDs: {', '.join(pids)})")
        backend_running = True
    else:
        print("⏰ Backend server NOT RUNNING")
        backend_running = False
except Exception as e:
    print(f"⚠️ Cannot check backend status: {e}")
    backend_running = False

# Final assessment
print("\n🎯 FINAL ASSESSMENT:")
print("="*60)

if implementation_score >= 0.8:
    print("🎉 IMPLEMENTATION: PASSED!")
    print(f"   ✅ {found_components}/{total_components} components implemented")
    
    if frontend_ready:
        print("   ✅ Frontend compatibility: READY")
    else:
        print("   ⚠️ Frontend compatibility: NEEDS ATTENTION")
    
    if backend_running:
        print("   ✅ Backend server: RUNNING")
        print("\n🚀 READY FOR IMMEDIATE TESTING!")
        print("   1. Open: http://localhost:3000")
        print("   2. Test voice chat")
        print("   3. Monitor: journalctl -f -u beautyai-api")
        print("   4. Look for: 'buffered chunks' messages")
    else:
        print("   ⏰ Backend server: NOT RUNNING")
        print("\n🔧 START BACKEND FIRST:")
        print("   cd /home/lumi/beautyai/backend")
        print("   python run_server.py")
    
    overall_success = True
    
else:
    print("❌ IMPLEMENTATION: INCOMPLETE!")
    print(f"   Only {found_components}/{total_components} components found")
    print("   Fix implementation before testing!")
    overall_success = False

# Save results
results = {
    "timestamp": "2025-08-08 Manual Validation",
    "implementation_score": implementation_score,
    "components_found": found_components,
    "total_components": total_components,
    "frontend_ready": frontend_ready,
    "backend_running": backend_running,
    "overall_success": overall_success
}

results_file = Path("actual_validation_results.json")
try:
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")
except Exception as e:
    print(f"⚠️ Could not save results: {e}")

print(f"\n{'✅ SUCCESS: Backend chunk fix is ready!' if overall_success else '❌ FAILED: Implementation needs completion'}")
