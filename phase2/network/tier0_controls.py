"""
tier0_controls.py -- the history-free / current-free controls for the Tier 0 sweep, resumable.

Which input carries the sweep's edge: the company's filing history, or the 63 current inputs?
Every job uses the config of sweep runs 48 and 72 (TASK = "direction", USE_TIME_CYCLES,
CAUSAL_READOUT and CLASS_WEIGHTED_LOSS all True, 15 epochs, same split) and changes only:

  job   HISTORY_MODE  CURRENT_MODE  DROP_LOOKAHEAD_FEATURES  seeds
  e     full          full          False                    0      reproduction of sweep run 48 (51.74%)
  d     none          none          False                    0      sanity: should be about 50%
  a     none          full          False                    0, 1   history-free
  b     none          full          True                     0, 1   history-free, look-ahead columns dropped
  c     full          none          False                    0, 1   history-only

e and d run first, so a failed reproduction shows up before the other six jobs.

Run / resume:  python tier0_controls.py           (skips every job that already has a results file)
Progress:      python tier0_controls.py --status
Stop:          create sweep/controls/STOP; it stops before the next job and deletes STOP.

Writes, in sweep/controls/ (the sweep's own results and log are never touched):
    results/<job_id>.json, logs/<job_id>.log, overrides/<job_id>.json
    tier0_controls_log.csv   the columns of tier0_sweep_log.csv plus HISTORY_MODE and CURRENT_MODE
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from tier0_sweep import JOB_TIMEOUT_HOURS, LOG_COLUMNS, SETTINGS, SWEEP_DIR, TRAIN_SCRIPT, check_pickle, stderr_tail

CONTROLS_DIR = SWEEP_DIR / "controls"
COLUMNS = LOG_COLUMNS[:4 + len(SETTINGS)] + ["HISTORY_MODE", "CURRENT_MODE"] + LOG_COLUMNS[4 + len(SETTINGS):]
FIXED = {"TASK": "direction", "USE_TIME_CYCLES": True, "CAUSAL_READOUT": True, "CLASS_WEIGHTED_LOSS": True, "EPOCHS": 15}
#             (label, HISTORY_MODE, CURRENT_MODE, DROP_LOOKAHEAD_FEATURES, SEED), in run order
GRID = [("e", "full", "full", False, 0), ("d", "none", "none", False, 0),
        ("a", "none", "full", False, 0), ("b", "none", "full", True, 0), ("c", "full", "none", False, 0),
        ("a", "none", "full", False, 1), ("b", "none", "full", True, 1), ("c", "full", "none", False, 1)]


def build_jobs():
    jobs = []
    for label, history, current, drop, seed in GRID:
        job_id = f"control_{label}_hist-{history}_cur-{current}_drop{int(drop)}_seed{seed}"
        jobs.append({"job_id": job_id, "overrides": {**FIXED, "HISTORY_MODE": history, "CURRENT_MODE": current,
                                                     "DROP_LOOKAHEAD_FEATURES": drop, "SEED": seed}})
    return jobs


def paths():
    p = {k: CONTROLS_DIR / k for k in ("results", "logs", "overrides")}
    for folder in p.values():
        folder.mkdir(parents=True, exist_ok=True)
    return p | {"log_csv": CONTROLS_DIR / "tier0_controls_log.csv", "stop": CONTROLS_DIR / "STOP"}


def done(job, p):
    return (p["results"] / f"{job['job_id']}.json").exists()


def append_row(p, row):
    new = not p["log_csv"].exists()
    with open(p["log_csv"], "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})


def next_run_number(p):
    """Controls are numbered C1, C2, ... so they never clash with the sweep's run numbers."""
    if not p["log_csv"].exists():
        return "C1"
    with open(p["log_csv"], newline="", encoding="utf-8") as f:
        return f"C{len(list(csv.DictReader(f))) + 1}"


def run_job(job, p):
    """One job in its own process, exactly like tier0_sweep.run_job. Raises KeyboardInterrupt after logging it."""
    job_id = job["job_id"]
    override_path, result_path, log_path = (p["overrides"] / f"{job_id}.json", p["results"] / f"{job_id}.json",
                                            p["logs"] / f"{job_id}.log")
    override_path.write_text(json.dumps(job["overrides"], indent=2), encoding="utf-8")
    env = {**os.environ, "TIER0_OVERRIDE_JSON": str(override_path), "RESULTS_PATH": str(result_path), "PYTHONUNBUFFERED": "1"}
    row = {"run": next_run_number(p), "job_id": job_id, **{k: job["overrides"][k] for k in SETTINGS},
           "HISTORY_MODE": job["overrides"]["HISTORY_MODE"], "CURRENT_MODE": job["overrides"]["CURRENT_MODE"]}
    interrupted = False
    with open(log_path, "w", encoding="utf-8") as out:
        process = subprocess.Popen([sys.executable, str(TRAIN_SCRIPT)], cwd=TRAIN_SCRIPT.parent, env=env, stdout=out,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        try:
            _, stderr = process.communicate(timeout=JOB_TIMEOUT_HOURS * 3600)
            reason = None if process.returncode == 0 else f"exit code {process.returncode}"
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
            reason = f"timeout after {JOB_TIMEOUT_HOURS} h"
        except KeyboardInterrupt:
            process.kill()
            _, stderr = process.communicate()
            reason, interrupted = "KeyboardInterrupt", True
        out.write("\n---------------- stderr\n" + stderr)
    if reason is None and not result_path.exists():
        reason = "finished without a results file"
    row["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if reason:
        row.update(status=f"CRASHED ({reason}) - no result", notes=stderr_tail(stderr))
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row.update(status="OK", **result["metrics"], runtime_s=round(result["runtime_s"], 1),
                   epochs_completed=result["epochs_completed"])
    append_row(p, row)
    if interrupted:
        raise KeyboardInterrupt
    return row


def main():
    parser = argparse.ArgumentParser(description="Run or resume the Tier 0 history/current controls.")
    parser.add_argument("--status", action="store_true", help="print progress only; no training")
    args = parser.parse_args()
    jobs, p = build_jobs(), paths()
    remaining = [j for j in jobs if not done(j, p)]
    print(f"{len(jobs) - len(remaining)} of {len(jobs)} control jobs done, {len(remaining)} to run")
    if args.status:
        for j in jobs:
            print(f"  {'done' if done(j, p) else '    '}  {j['job_id']}")
        return
    check_pickle()
    try:
        for job in remaining:
            if p["stop"].exists():
                p["stop"].unlink()
                print("STOP file found: stopping before the next job (STOP removed, so the next run resumes).")
                return
            print(f"[{datetime.now():%Y-%m-%d %H:%M}] job #{jobs.index(job) + 1}/{len(jobs)} {job['job_id']} ...", flush=True)
            started = time.time()
            row = run_job(job, p)
            extra = (f"  balanced accuracy {float(row['balanced_accuracy_model']):.4f}" if row["status"] == "OK" else f": {row['notes']}")
            print(f"    {row['status']} in {(time.time() - started) / 60:.1f} min{extra}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. The running job is logged as CRASHED (KeyboardInterrupt) and runs again on resume.")
        return
    left = sum(not done(j, p) for j in jobs)
    print(f"controls pass finished: {len(jobs) - left} of {len(jobs)} done" + (f", {left} crashed (rerun this script)" if left else ""))


if __name__ == "__main__":
    main()
