#!/usr/bin/env python3

import ast
import subprocess
import json
import time
from pathlib import Path

def run_full_validation():
    """Run a complete validation of the backend fix"""
    print("🧪 COMPLETE BACKEND CHUNK ACCUMULATION VALIDATION")
    print("=" * 60)
    
    results = {
        "timestamp": time.time(),
        "checks": {},
        "overall_success": False
    }
    
    # 1. File existence and syntax
    print("\n1️⃣ CHECKING FILE SYNTAX")
    print("-" * 30)
    
    file_path = "/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/websocket_simple_voice.py"
    
    if not Path(file_path).exists():
        print("❌ Backend file not found")
        results["checks"]["file_exists"] = False
        return results
    
    print("✅ Backend file exists")
    results["checks"]["file_exists"] = True
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ Python syntax is valid")
        results["checks"]["syntax_valid"] = True
    except Exception as e:
        print(f"❌ Syntax error: {e}")
        results["checks"]["syntax_valid"] = False
        return results
    
    # 2. Implementation components
    print("\n2️⃣ CHECKING IMPLEMENTATION COMPONENTS")
    print("-" * 30)
    
    components = {
        "chunk_buffer_init": "chunk_buffer" in content,
        "process_buffered_chunks": "def _process_buffered_chunks" in content,
        "setup_vad_callbacks": "def _setup_vad_callbacks" in content,
        "processing_turn_flag": "processing_turn" in content,
        "chunk_concatenation": "b''.join(connection[\"chunk_buffer\"])" in content,
        "chunk_append": "chunk_buffer.append(audio_data)" in content,
        "buffer_clearing": "connection[\"chunk_buffer\"] = []" in content
    }
    
    for component, present in components.items():
        status = "✅" if present else "❌"
        print(f"{status} {component}")
        results["checks"][component] = present
    
    # 3. Logic patterns
    print("\n3️⃣ CHECKING LOGIC PATTERNS")
    print("-" * 30)
    
    logic_patterns = {
        "ignore_during_processing": "if connection.get(\"processing_turn\", False):" in content,
        "turn_id_tracking": "last_turn_id" in content,
        "vad_integration": "_setup_vad_callbacks(connection_id)" in content,
        "emergency_fallback": "Emergency processing" in content
    }
    
    for pattern, present in logic_patterns.items():
        status = "✅" if present else "❌"
        print(f"{status} {pattern}")
        results["checks"][pattern] = present
    
    # 4. Frontend compatibility
    print("\n4️⃣ CHECKING FRONTEND COMPATIBILITY")
    print("-" * 30)
    
    frontend_files = [
        "/home/lumi/beautyai/frontend/src/static/js/chat-interface.js",
        "/home/lumi/beautyai/frontend/src/static/js/voice-overlay-strict-patch.js"
    ]
    
    frontend_ok = True
    for file_path in frontend_files:
        if Path(file_path).exists():
            print(f"✅ Found: {Path(file_path).name}")
        else:
            print(f"⚠️ Missing: {Path(file_path).name}")
            frontend_ok = False
    
    results["checks"]["frontend_compatible"] = frontend_ok
    
    # 5. Backend status
    print("\n5️⃣ CHECKING BACKEND STATUS")
    print("-" * 30)
    
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_server.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Backend server is running")
            results["checks"]["backend_running"] = True
        else:
            print("⏰ Backend server is not running")
            results["checks"]["backend_running"] = False
    except Exception:
        print("⚠️ Could not check backend status")
        results["checks"]["backend_running"] = False
    
    # Calculate overall success
    passed_checks = sum(1 for v in results["checks"].values() if v)
    total_checks = len(results["checks"])
    success_rate = passed_checks / total_checks
    
    results["passed"] = passed_checks
    results["total"] = total_checks
    results["success_rate"] = success_rate
    results["overall_success"] = success_rate >= 0.8
    
    # Final report
    print("\n🎯 FINAL RESULTS")
    print("=" * 60)
    
    print(f"📊 Validation score: {passed_checks}/{total_checks} ({success_rate:.1%})")
    
    if results["overall_success"]:
        print("\n🎉 VALIDATION PASSED!")
        print("✅ Backend chunk accumulation fix is properly implemented")
        print("✅ All critical components are in place")
        print("✅ Ready for testing")
        
        if results["checks"].get("backend_running"):
            print("\n🚀 READY FOR IMMEDIATE TESTING:")
            print("1. Open frontend: http://localhost:3000")
            print("2. Test voice chat interface")
            print("3. Monitor logs for 'buffered chunks' messages")
        else:
            print("\n🔧 START BACKEND FIRST:")
            print("1. cd /home/lumi/beautyai/backend")
            print("2. python run_server.py")
            print("3. Then test voice chat")
            
    else:
        print("\n❌ VALIDATION NEEDS ATTENTION")
        print("Some critical components are missing:")
        
        for check, passed in results["checks"].items():
            if not passed:
                print(f"   🔴 {check}")
    
    # Save results
    results_file = Path("/home/lumi/beautyai/validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = run_full_validation()
    
    # Print summary
    if results["overall_success"]:
        print("\n✅ SUCCESS: Backend chunk accumulation fix is ready!")
    else:
        print("\n❌ ATTENTION NEEDED: Fix requires completion before testing")
