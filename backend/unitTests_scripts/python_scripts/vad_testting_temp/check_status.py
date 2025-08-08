#!/usr/bin/env python3

import subprocess
import json
import time
from pathlib import Path

def check_backend_status():
    """Check if backend is running and accessible"""
    print("🔍 CHECKING BACKEND STATUS")
    print("=" * 40)
    
    # Check if backend process is running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_server.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ Backend process running (PIDs: {', '.join(pids)})")
            running = True
        else:
            print("⏰ Backend server is not running")
            running = False
    except Exception as e:
        print(f"⚠️ Could not check backend status: {e}")
        running = False
    
    # Check if port 8000 is in use
    try:
        result = subprocess.run(
            ["netstat", "-ln", "|", "grep", ":8000"],
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout:
            print("✅ Port 8000 is in use")
        else:
            print("⏰ Port 8000 is not in use")
    except Exception:
        print("⚠️ Could not check port status")
    
    # Try to connect to health endpoint
    try:
        import urllib.request
        import urllib.error
        
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as response:
                if response.status == 200:
                    print("✅ Backend health endpoint responding")
                    return True
        except urllib.error.URLError:
            print("❌ Backend health endpoint not responding")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            
    except ImportError:
        print("⚠️ Cannot import urllib for health check")
    
    return running

def create_quick_test_plan():
    """Create a quick test plan"""
    print("\n📋 QUICK TEST PLAN")
    print("=" * 40)
    
    backend_running = check_backend_status()
    
    if backend_running:
        print("\n🚀 BACKEND IS RUNNING - Ready for testing!")
        print("1. Open frontend: http://localhost:3000")
        print("2. Test voice chat interface")
        print("3. Monitor logs: journalctl -f -u beautyai-api")
        print("4. Look for 'buffered chunks' messages")
        print("5. Verify no duplicate responses")
    else:
        print("\n⏳ BACKEND NOT RUNNING - Start it first:")
        print("1. cd /home/lumi/beautyai/backend")
        print("2. python run_server.py")
        print("3. Wait for 'Server started' message")
        print("4. Then test voice chat")
    
    # Check if validation results exist
    results_file = Path("/home/lumi/beautyai/validation_results.json")
    if results_file.exists():
        print(f"\n📄 Previous validation results found: {results_file}")
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            print(f"   Timestamp: {time.ctime(data.get('timestamp', 0))}")
            print(f"   Success: {'✅' if data.get('success') else '❌'}")
            print(f"   Passed: {data.get('passed', 0)}/{data.get('total', 0)} checks")
        except Exception:
            print("   ⚠️ Could not read validation results")
    else:
        print("\n📄 No previous validation results found")
    
    return backend_running

if __name__ == "__main__":
    create_quick_test_plan()
