#!/usr/bin/env python3
"""
BeautyAI Independent Benchmarking & Testing System Test Runner

Comprehensive test runner for the independent benchmarking and testing system
with organized test categories and detailed reporting.
"""
import os
import sys
import subprocess
import argparse
import time
from typing import List, Dict, Optional
from pathlib import Path

# Set up path for independent system
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))


class IndependentTestRunner:
    """Test runner for independent benchmarking & testing system."""
    
    def __init__(self, system_root: str = None):
        """Initialize test runner."""
        if system_root is None:
            # Get the system root from the script location
            script_dir = Path(__file__).parent.resolve()
            system_root = script_dir
        
        self.system_root = Path(system_root).resolve()
        self.tests_dir = self.system_root / "tests"
        
        # Test categories for independent system
        self.test_categories = {
            "benchmarking": [
                "test_enhanced_benchmarking.py"
            ],
            "content_filter": [
                "test_content_filter.py"
            ],
            "all": [
                "test_enhanced_benchmarking.py",
                "test_content_filter.py"
            ]
        }
    
    def run_category_tests(self, category: str, verbose: bool = False) -> Dict:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            print(f"❌ Unknown test category: {category}")
            print(f"📋 Available categories: {', '.join(self.test_categories.keys())}")
            return {"success": False, "error": f"Unknown category: {category}"}
        
        print(f"🧪 Running {category.upper()} tests...")
        print("=" * 60)
        
        test_files = self.test_categories[category]
        results = {
            "category": category,
            "total_files": len(test_files),
            "passed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "test_results": [],
            "total_time": 0
        }
        
        start_time = time.time()
        
        for test_file in test_files:
            test_path = self.tests_dir / test_file
            
            if not test_path.exists():
                print(f"⚠️  Test file not found: {test_file}")
                results["skipped_files"] += 1
                results["test_results"].append({
                    "file": test_file,
                    "status": "skipped",
                    "reason": "File not found"
                })
                continue
            
            print(f"\n📂 Running: {test_file}")
            file_result = self._run_single_test_file(test_path, verbose)
            results["test_results"].append(file_result)
            
            if file_result["status"] == "passed":
                results["passed_files"] += 1
                print(f"✅ {test_file} - PASSED")
            else:
                results["failed_files"] += 1
                print(f"❌ {test_file} - FAILED")
                if file_result.get("error"):
                    print(f"   Error: {file_result['error']}")
        
        results["total_time"] = time.time() - start_time
        
        # Print category summary
        print(f"\n📊 {category.upper()} Test Results Summary:")
        print(f"   ✅ Passed: {results['passed_files']}")
        print(f"   ❌ Failed: {results['failed_files']}") 
        print(f"   ⚠️  Skipped: {results['skipped_files']}")
        print(f"   ⏱️  Total time: {results['total_time']:.2f}s")
        
        return results
    
    def _run_single_test_file(self, test_path: Path, verbose: bool = False) -> Dict:
        """Run a single test file."""
        try:
            cmd = [sys.executable, "-m", "pytest", str(test_path)]
            
            if verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")
            
            # Add coverage if available
            try:
                subprocess.run([sys.executable, "-c", "import pytest_cov"], 
                             check=True, capture_output=True)
                cmd.extend(["--cov=beautyai_inference", "--cov-report=term-missing"])
            except subprocess.CalledProcessError:
                pass  # pytest-cov not available
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.system_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test file
            )
            execution_time = time.time() - start_time
            
            return {
                "file": test_path.name,
                "status": "passed" if result.returncode == 0 else "failed",
                "execution_time": execution_time,
                "stdout": result.stdout if verbose else "",
                "stderr": result.stderr if result.returncode != 0 else "",
                "error": result.stderr.split('\n')[-2] if result.returncode != 0 and result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "file": test_path.name,
                "status": "failed",
                "execution_time": 300,
                "error": "Test execution timed out (5 minutes)"
            }
        except Exception as e:
            return {
                "file": test_path.name,
                "status": "failed", 
                "execution_time": 0,
                "error": str(e)
            }
    
    def run_all_tests(self, verbose: bool = False) -> Dict:
        """Run all test categories."""
        print("🚀 Running ALL BeautyAI Framework Tests")
        print("=" * 80)
        
        overall_results = {
            "categories": {},
            "total_passed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "total_time": 0,
            "success": True
        }
        
        start_time = time.time()
        
        for category in self.test_categories.keys():
            if category == "integration":
                continue  # Skip integration as it overlaps with other categories
            
            category_results = self.run_category_tests(category, verbose)
            overall_results["categories"][category] = category_results
            
            overall_results["total_passed"] += category_results["passed_files"]
            overall_results["total_failed"] += category_results["failed_files"] 
            overall_results["total_skipped"] += category_results["skipped_files"]
            
            if category_results["failed_files"] > 0:
                overall_results["success"] = False
        
        overall_results["total_time"] = time.time() - start_time
        
        # Print overall summary
        print("\n" + "=" * 80)
        print("🏆 OVERALL TEST RESULTS SUMMARY")
        print("=" * 80)
        
        for category, results in overall_results["categories"].items():
            status = "✅" if results["failed_files"] == 0 else "❌"
            print(f"{status} {category.upper()}: {results['passed_files']}/{results['total_files']} passed")
        
        print(f"\n📊 Total Results:")
        print(f"   ✅ Total Passed: {overall_results['total_passed']}")
        print(f"   ❌ Total Failed: {overall_results['total_failed']}")
        print(f"   ⚠️  Total Skipped: {overall_results['total_skipped']}")
        print(f"   ⏱️  Total Time: {overall_results['total_time']:.2f}s")
        
        if overall_results["success"]:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"\n💥 SOME TESTS FAILED! 💥")
            print("   Please check the detailed output above for specific failures.")
        
        return overall_results
    
    def run_specific_tests(self, test_files: List[str], verbose: bool = False) -> Dict:
        """Run specific test files."""
        print(f"🎯 Running specific tests: {', '.join(test_files)}")
        print("=" * 60)
        
        results = {
            "test_files": test_files,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "results": []
        }
        
        for test_file in test_files:
            test_path = self.tests_dir / test_file
            
            if not test_path.exists():
                print(f"⚠️  Test file not found: {test_file}")
                results["skipped"] += 1
                continue
            
            file_result = self._run_single_test_file(test_path, verbose)
            results["results"].append(file_result)
            
            if file_result["status"] == "passed":
                results["passed"] += 1
                print(f"✅ {test_file} - PASSED")
            else:
                results["failed"] += 1
                print(f"❌ {test_file} - FAILED")
        
        return results
    
    def list_available_tests(self):
        """List all available tests by category."""
        print("📋 Available Test Categories and Files:")
        print("=" * 60)
        
        for category, test_files in self.test_categories.items():
            print(f"\n📁 {category.upper()}:")
            for test_file in test_files:
                test_path = self.tests_dir / test_file
                status = "✅" if test_path.exists() else "❌ (missing)"
                print(f"   - {test_file} {status}")
    
    def check_test_environment(self):
        """Check if test environment is properly set up."""
        print("🔍 Checking Test Environment...")
        print("=" * 60)
        
        checks = {
            "Python version": sys.version.split()[0],
            "System root exists": self.system_root.exists(),
            "Tests directory exists": self.tests_dir.exists(),
            "BeautyAI package importable": self._check_import("beautyai_inference"),
            "Pytest available": self._check_import("pytest"),
            "Required dependencies": self._check_dependencies()
        }
        
        for check, result in checks.items():
            if isinstance(result, bool):
                status = "✅" if result else "❌"
                print(f"{status} {check}")
            else:
                print(f"ℹ️  {check}: {result}")
        
        return all(isinstance(v, bool) and v for v in checks.values() if isinstance(v, bool))
    
    def _check_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _check_dependencies(self) -> str:
        """Check required dependencies."""
        required = ["requests", "click", "torch", "transformers"]
        available = []
        missing = []
        
        for dep in required:
            if self._check_import(dep):
                available.append(dep)
            else:
                missing.append(dep)
        
        if missing:
            return f"{len(available)}/{len(required)} (missing: {', '.join(missing)})"
        else:
            return f"{len(available)}/{len(required)} ✅"


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="BeautyAI Framework Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                     # Run all tests
  python run_tests.py --category cli            # Run CLI tests only
  python run_tests.py --category content_filter # Run content filter tests
  python run_tests.py --list                    # List available tests
  python run_tests.py --check                   # Check test environment
  python run_tests.py --files test_cli.py       # Run specific test file
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all tests")
    group.add_argument("--category", help="Run tests for specific category")
    group.add_argument("--files", nargs="+", help="Run specific test files")
    group.add_argument("--list", action="store_true", help="List available tests")
    group.add_argument("--check", action="store_true", help="Check test environment")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IndependentTestRunner(args.project_root)
    
    if args.check:
        env_ok = runner.check_test_environment()
        sys.exit(0 if env_ok else 1)
    
    if args.list:
        runner.list_available_tests()
        sys.exit(0)
    
    # Run tests based on arguments
    if args.all:
        results = runner.run_all_tests(args.verbose)
        sys.exit(0 if results["success"] else 1)
    
    elif args.category:
        results = runner.run_category_tests(args.category, args.verbose)
        sys.exit(0 if results["failed_files"] == 0 else 1)
    
    elif args.files:
        results = runner.run_specific_tests(args.files, args.verbose)
        sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
