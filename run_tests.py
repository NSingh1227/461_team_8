"""
Test runner script for Milestone 1 validation.
Executes unit tests and reports coverage as specified in requirements.
"""

import subprocess
import sys
import os


def run_tests():
    """
    Run the test suite and report results in the required format.
    
    Expected output format: "X/Y test cases passed. Z% line coverage achieved."
    """
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--tb=short",
            "-v"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        print("=== Test Execution Output ===")
        print(result.stdout)

        if result.stderr:
            print("=== Errors/Warnings ===")
            print(result.stderr)

        # Parse results for summary format
        lines = result.stdout.split('\n')

        test_results = None
        coverage_percent = None

        for line in lines:
            # Handle "X failed, Y passed in Zs"
            if "failed" in line and "passed" in line and "in" in line:
                parts = line.split()
                failed_idx = parts.index("failed,")
                passed_idx = parts.index("passed")
                failed_count = int(parts[failed_idx - 1])
                passed_count = int(parts[passed_idx - 1])
                total_count = failed_count + passed_count
                test_results = f"{passed_count}/{total_count}"

            # Handle "X passed in Zs"
            elif "passed" in line and "in" in line and line.strip().startswith("="):
                # Example: "============================== 17 passed in 0.28s =============================="
                parts = line.split()
                passed_idx = parts.index("passed")
                passed_count = int(parts[passed_idx - 1])
                test_results = f"{passed_count}/{passed_count}"

            # Handle coverage line
            if line.strip().startswith("TOTAL") and "%" in line:
                parts = line.split()
                # last column should be like "90%"
                coverage_percent = parts[-1].rstrip("%")

        # Print required format
        print("\n=== MILESTONE 1 TEST SUMMARY ===")
        if test_results and coverage_percent:
            print(f"{test_results} test cases passed. {coverage_percent}% line coverage achieved.")
        else:
            print("Could not parse test results. Check output above.")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)