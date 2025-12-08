import sys
import subprocess
import os
import glob
from pathlib import Path

def main():
    # 1. CLEANUP: Remove old coverage data to ensure a fresh start
    if os.path.exists(".coverage"):
        os.remove(".coverage")

    # 2. DISCOVERY: Find all test files and sort them to Ensure Order
    test_files = sorted(glob.glob("tests/**/test_rdeps_*.py", recursive=True))
    
    if not test_files:
        print("No test files found!")
        sys.exit(1)

    print(f"Found {len(test_files)} test files. Running sequentially with isolation...")

    # 3. EXECUTION LOOP: Run each file in a separate process
    failure_occurred = False
    
    for test_file in test_files:
        print(f"--> Running {test_file}")
        
        # Construct the command
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,                     # Run specific file
            "-v",
            "-m", "rdeps",                 # Your marker
            "--cov-config=.coveragerc-r-dependencies",
            "--cov-append",                # CRITICAL: Append to .coverage file instead of overwriting
            # Note: We delay reporting (xml/json/term) until the very end for speed
        ]

        # Run the command
        result = subprocess.run(cmd)
        
        # If any file fails, mark failure but continue (or break if you want fail-fast)
        if result.returncode != 0:
            failure_occurred = True
            # break  # Uncomment this line if you want to stop immediately on error

    # 4. REPORTING: Generate the reports once at the end
    print("\n--> Generating Coverage Reports...")
    
    # Generate Terminal Report
    subprocess.run([
        sys.executable, "-m", "coverage", "report", 
        "--rcfile=.coveragerc-r-dependencies"
    ])
    
    # Generate XML Report
    subprocess.run([
        sys.executable, "-m", "coverage", "xml", 
        "--rcfile=.coveragerc-r-dependencies"
    ])

    # Generate JSON Report
    subprocess.run([
        sys.executable, "-m", "coverage", "json", 
        "--rcfile=.coveragerc-r-dependencies"
    ])

    # Exit with non-zero code if any tests failed
    if failure_occurred:
        sys.exit(1)

if __name__ == "__main__":
    main()