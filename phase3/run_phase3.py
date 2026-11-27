"""
Runs every Phase 3 step in order, with a banner and a timing for each.

  1. data_phase3/association_events.py   Task 1 numbers, CSVs and T1_Association_* sheets
  2. data_phase3/sequential_events.py    Task 2 numbers, CSVs and T2_Sequential_* sheets
  3. charts3/association_charts.py       charts 01-04
  4. charts3/sequential_charts.py        charts 05-07

Run from anywhere:  ..\\.venv\\Scripts\\python.exe run_phase3.py
Pass --fresh to delete the generated workbook first, so it is rebuilt from both scripts.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("data_phase3", "association_events.py", "Task 1: itemsets, rules, Rules A and B, then the T1 sheets (about 5 min, mostly the Excel save)"),
    ("data_phase3", "sequential_events.py", "Task 2: transitions, patterns, robustness, shuffle check, then the T2 sheets (about 6 min, mostly the Excel save)"),
    ("charts3", "association_charts.py", "Charts 01-04 (seconds)"),
    ("charts3", "sequential_charts.py", "Charts 05-07 (seconds)"),
]


def main():
    if "--fresh" in sys.argv:
        workbook = HERE / "data_phase3" / "processed" / "220227286.xlsx"
        workbook.unlink(missing_ok=True)
        print(f"--fresh: deleted {workbook.name}; both scripts rebuild it.")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf8"}
    started = time.time()
    for number, (folder, script, what) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 78}\n[{number}/{len(STEPS)}] {folder}/{script}\n      {what}\n{'=' * 78}", flush=True)
        step_start = time.time()
        result = subprocess.run([sys.executable, script], cwd=HERE / folder, env=env)
        minutes, seconds = divmod(time.time() - step_start, 60)
        if result.returncode != 0:
            print(f"\n[{number}/{len(STEPS)}] FAILED after {minutes:.0f}m {seconds:.0f}s (exit code {result.returncode}). Later steps were not run.")
            sys.exit(result.returncode)
        print(f"\n[{number}/{len(STEPS)}] finished in {minutes:.0f}m {seconds:.0f}s", flush=True)
    minutes, seconds = divmod(time.time() - started, 60)
    print(f"\nALL {len(STEPS)} STEPS DONE in {minutes:.0f}m {seconds:.0f}s")


if __name__ == "__main__":
    main()
