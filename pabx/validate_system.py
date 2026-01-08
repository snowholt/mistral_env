#!/usr/bin/env python3
"""
System validation script
Check that all components are properly configured
"""

import sys
from pathlib import Path
import importlib.util

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_mark(success: bool) -> str:
    return f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"

def main():
    print("=" * 60)
    print("BeautyAI PABX System Validation")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Check directory structure
    print("📁 Directory Structure")
    required_dirs = [
        "src/core/sip",
        "src/core/rtp",
        "src/modules/audio",
        "src/modules/sniffer",
        "src/modules/ht813",
        "src/services",
        "src/api",
        "src/utils",
        "config",
        "logs",
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists() and path.is_dir()
        print(f"  {check_mark(exists)} {dir_path}")
        if not exists:
            all_passed = False
    
    print()
    
    # Check configuration files
    print("⚙️  Configuration Files")
    config_files = [
        "config/settings.yaml",
        "config/devices.json",
    ]
    
    for file_path in config_files:
        path = Path(file_path)
        exists = path.exists() and path.is_file()
        print(f"  {check_mark(exists)} {file_path}")
        if not exists:
            all_passed = False
    
    print()
    
    # Check Python modules
    print("🐍 Python Modules")
    modules = [
        ("yaml", "PyYAML"),
        ("pyaudio", "PyAudio"),
        ("scapy", "scapy"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("soundfile", "soundfile"),
        ("requests", "requests"),
        ("bs4", "beautifulsoup4"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "uvicorn"),
        ("click", "click"),
        ("rich", "rich"),
    ]
    
    for module_name, package_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  {check_mark(True)} {package_name}")
        except ImportError:
            print(f"  {check_mark(False)} {package_name} - NOT INSTALLED")
            all_passed = False
    
    print()
    
    # Check core modules can be imported
    print("📦 Core Modules Import")
    core_imports = [
        "src.utils.config",
        "src.utils.logger",
        "src.core.sip.parser",
        "src.core.sip.builder",
        "src.core.rtp.packet",
        "src.core.rtp.stream",
        "src.modules.audio.codecs",
        "src.modules.sniffer.capture",
        "src.services.sip_server",
        "src.services.rtp_handler",
        "src.services.call_manager",
        "src.api.server",
    ]
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    for module_path in core_imports:
        try:
            module = importlib.import_module(module_path)
            print(f"  {check_mark(True)} {module_path}")
        except Exception as e:
            print(f"  {check_mark(False)} {module_path} - ERROR: {e}")
            all_passed = False
    
    print()
    
    # Check executable scripts
    print("📜 Executable Scripts")
    scripts = [
        "run_server.py",
        "install.sh",
    ]
    
    for script in scripts:
        path = Path(script)
        exists = path.exists()
        executable = path.stat().st_mode & 0o111 if exists else False
        status = exists and executable
        print(f"  {check_mark(status)} {script} {'(executable)' if status else '(not executable)'}")
        if not status:
            all_passed = False
    
    print()
    
    # Check service files
    print("🔧 Service Files")
    services = [
        "pabx-server.service",
        "pabx-sniffer.service",
    ]
    
    for service in services:
        path = Path(service)
        exists = path.exists() and path.is_file()
        print(f"  {check_mark(exists)} {service}")
        if not exists:
            all_passed = False
    
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print(f"{GREEN}✅ All checks passed!{RESET}")
        print()
        print("System is ready for deployment.")
        print("Next steps:")
        print("  1. Run: ./install.sh")
        print("  2. Configure: edit config/settings.yaml")
        print("  3. Start: ./run_server.py --mode api")
        return 0
    else:
        print(f"{RED}❌ Some checks failed!{RESET}")
        print()
        print("Please fix the issues above and run this script again.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
