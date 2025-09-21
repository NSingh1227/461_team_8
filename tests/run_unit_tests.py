#!/usr/bin/env python3
"""
Run all unit tests with coverage analysis.
Quick script for running just the unit tests without the full integration suite.
"""

import subprocess
import sys
import os

def run_unit_tests():
    """Run all unit tests with coverage."""
    print("Running All Unit Tests with Coverage")
    print("=" * 50)
    
    unit_tests = [
        "tests/unit/test_busfactor.py",
        "tests/unit/test_license.py", 
        "tests/unit/test_url_processor.py",
        "tests/unit/test_dataset_code.py",
        "tests/unit/test_dataset_quality.py"
    ]
    
    # Clear previous coverage
    subprocess.run(["python", "-m", "coverage", "erase"], capture_output=True)
    
    # Run each unit test
    total_tests = 0
    for i, test_file in enumerate(unit_tests):
        print(f"\n[{i+1}/{len(unit_tests)}] Running {test_file}")
        
        if i == 0:
            # First test - start coverage
            result = subprocess.run(["python", "-m", "coverage", "run", test_file], 
                                  capture_output=True, text=True)
        else:
            # Subsequent tests - append to coverage
            result = subprocess.run(["python", "-m", "coverage", "run", "--append", test_file], 
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count tests from output
            if "Ran" in result.stderr:
                test_count = result.stderr.split("Ran ")[1].split(" test")[0]
                total_tests += int(test_count)
                print(f"✅ {test_count} tests passed")
            else:
                print("✅ Tests passed")
        else:
            print(f"❌ Tests failed: {result.stderr}")
    
    # Generate coverage report
    print(f"\n📊 Coverage Report")
    print("=" * 50)
    result = subprocess.run(["python", "-m", "coverage", "report"], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    print(f"\n🎯 Summary: {total_tests} unit tests completed across 5 test files")
    print("📈 All unit test files now included in coverage analysis")

if __name__ == "__main__":
    run_unit_tests()