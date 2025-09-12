"""
Test runner script for Milestone 1 and Phase 2 preparation.
Executes unit tests for metric calculators, results storage, and storage manager.
"""

import subprocess
import sys

def run_tests():
    """
    Run pytest and report results.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--maxfail=1", "--disable-warnings"],
            capture_output=False,
            check=True
        )
        if result.returncode == 0:
            print("All tests passed successfully!")
        else:
            print("Some tests failed.")
    except subprocess.CalledProcessError as e:
        print("Tests failed with error:", e)
        sys.exit(e.returncode)

if __name__ == "__main__":
    run_tests()
