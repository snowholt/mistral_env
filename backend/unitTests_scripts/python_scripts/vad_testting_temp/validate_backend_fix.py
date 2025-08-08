#!/usr/bin/env python3
"""
Simplified backend chunk accumulation validation.
This test checks the backend logs and endpoint structure to validate the fix.
"""

import sys
import time
import subprocess
import json
from pathlib import Path

def check_backend_chunk_fix():
    """Check if the backend chunk accumulation fix is implemented"""
    print("🔍 Validating backend chunk accumulation fix...")
    
    websocket_file = Path("/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py")
    
    if not websocket_file.exists():
        print("❌ WebSocket endpoint file not found")
        return False
    
    # Read the file and check for key fix components
    with open(websocket_file, 'r') as f:
        content = f.read()
    
    checks = {
        "chunk_buffer": "chunk_buffer" in content,
        "process_buffered_chunks": "_process_buffered_chunks" in content,
        "processing_turn": "processing_turn" in content,
        "webm_accumulation": "b''.join(connection[\"chunk_buffer\"])" in content,
        "vad_callbacks": "_setup_vad_callbacks" in content
    }
    
    print("\n📋 Backend Fix Components:")
    all_good = True
    for component, present in checks.items():
        status = "✅" if present else "❌"
        print(f"  {status} {component}")
        if not present:
            all_good = False
    
    return all_good

def check_backend_running():
    """Check if backend process is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_server.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Backend server is running")
            return True
        else:
            print("⏰ Backend server is not running")
            return False
    except Exception as e:
        print(f"⚠️ Could not check backend status: {e}")
        return False

def check_logs_for_duplicates():
    """Check recent logs for duplicate response patterns"""
    print("\n🔍 Checking recent logs for duplicate response patterns...")
    
    try:
        # Check journalctl for recent backend logs
        result = subprocess.run([
            "journalctl", "-u", "beautyai-api", "--since", "5 minutes ago", 
            "--no-pager", "-q"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            log_lines = result.stdout.split('\n')
            
            # Look for patterns indicating our fix
            chunk_processing = sum(1 for line in log_lines if "buffered chunks" in line.lower())
            duplicate_responses = sum(1 for line in log_lines if "duplicate" in line.lower())
            webm_errors = sum(1 for line in log_lines if "webm" in line.lower() and "error" in line.lower())
            
            print(f"📊 Log Analysis (last 5 minutes):")
            print(f"  📦 Chunk processing mentions: {chunk_processing}")
            print(f"  🔄 Duplicate mentions: {duplicate_responses}")
            print(f"  ❌ WebM errors: {webm_errors}")
            
            return {
                "chunk_processing": chunk_processing,
                "duplicate_responses": duplicate_responses,
                "webm_errors": webm_errors
            }
        else:
            print("⚠️ Could not access system logs")
            return None
            
    except Exception as e:
        print(f"⚠️ Log analysis failed: {e}")
        return None

def validate_fix_implementation():
    """Main validation function"""
    print("🧪 Backend Chunk Accumulation Fix Validation")
    print("=" * 50)
    
    # Check 1: Code implementation
    print("\n1️⃣ Checking code implementation...")
    code_ok = check_backend_chunk_fix()
    
    # Check 2: Backend status
    print("\n2️⃣ Checking backend status...")
    backend_running = check_backend_running()
    
    # Check 3: Log analysis
    print("\n3️⃣ Analyzing recent logs...")
    log_analysis = check_logs_for_duplicates()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    if code_ok:
        print("✅ Code implementation: COMPLETE")
        print("   - Chunk buffer system implemented")
        print("   - Buffered chunk processing added")
        print("   - Turn-based processing logic in place")
        print("   - VAD callback integration ready")
    else:
        print("❌ Code implementation: INCOMPLETE")
        print("   - Missing required fix components")
    
    if backend_running:
        print("✅ Backend status: RUNNING")
    else:
        print("⏰ Backend status: NOT RUNNING")
        print("   - Start backend to test chunk accumulation")
    
    if log_analysis:
        if log_analysis["webm_errors"] == 0:
            print("✅ Log analysis: NO WEBM ERRORS")
        else:
            print(f"⚠️ Log analysis: {log_analysis['webm_errors']} WebM errors detected")
    else:
        print("⚠️ Log analysis: UNAVAILABLE")
    
    # Overall assessment
    fix_deployed = code_ok
    ready_for_testing = code_ok and backend_running
    
    print("\n🎯 NEXT STEPS:")
    if fix_deployed and ready_for_testing:
        print("✅ Backend chunk accumulation fix is deployed and ready!")
        print("   - Test with frontend to validate end-to-end functionality")
        print("   - Monitor logs for chunk buffering messages")
        print("   - Verify no more duplicate responses")
    elif fix_deployed:
        print("⏳ Backend fix is deployed, start server to test:")
        print("   cd /home/lumi/beautyai/backend && python run_server.py")
    else:
        print("❌ Backend fix needs completion:")
        print("   - Review WebSocket endpoint implementation")
        print("   - Ensure all fix components are in place")

if __name__ == "__main__":
    validate_fix_implementation()
