"""
tier0_followups.py -- the Tier 0 follow-up jobs, resumable. Nothing earlier in sweep/ is read or written.

Every job: TASK = "direction", USE_TIME_CYCLES, CAUSAL_READOUT and CLASS_WEIGHTED_LOSS all True,
DROP_LOOKAHEAD_FEATURES False, 15 epochs, the paper's split; seeds 0 and 1 (seed 2 stays skipped).

  part_a  saved predictions for bootstrap_ci.py (SAVE_PREDICTIONS = True), 6 jobs:
          full model, no-history (HISTORY_MODE = "none"), history-only (CURRENT_MODE = "none").
          These are RERUNS of sweep runs 48/72 and controls C3/C6, C5/C8; GPU training is not
          bit-for-bit deterministic, so their scores differ slightly from the originals.
          -> sweep/predictions/<job_id>.csv, sweep/predictions/tier0_predictions_log.csv
  part_b  which current inputs carry the edge, on the no-history model, 24 jobs:
          drop one group (DROP_INPUT_GROUPS = [group]) for each of the 6 groups in
          tier0_train.INPUT_GROUPS, then keep one group (every other group dropped).
          The drop-one jobs run first; the keep-one jobs are the optional half.
          -> sweep/ablation/, sweep/ablation/tier0_ablation_log.csv

Run / resume:  python tier0_followups.py part_a          (skips every job that has a results file)
Progress:      python tier0_followups.py part_a --status
Stop:          create sweep/predictions/STOP or sweep/ablation/STOP; it stops before the next job.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from tier0_controls import COLUMNS as CONTROLS_COLUMNS
from tier0_sweep import JOB_TIMEOUT_HOURS, SETTINGS, SWEEP_DIR, TRAIN_SCRIPT, check_pickle, stderr_tail
from tier0_train import INPUT_GROUPS

FIXED = {"TASK": "direction", "USE_TIME_CYCLES": True, "CAUSAL_READOUT": True, "CLASS_WEIGHTED_LOSS": True,
         "DROP_LOOKAHEAD_FEATURES": False, "EPOCHS": 15}
SEEDS = [0, 1]
MODELS = {"full": {"HISTORY_MODE": "full", "CURRENT_MODE": "full"},
          "nohist": {"HISTORY_MODE": "none", "CURRENT_MODE": "full"},
          "histonly": {"HISTORY_MODE": "full", "CURRENT_MODE": "none"}}


def part_a_jobs():
    return [{"job_id": f"pred_{model}_seed{seed}", "overrides": {**FIXED, **modes, "SEED": seed, "SAVE_PREDICTIONS": True},
             "log": {"predictions_file": f"pred_{model}_seed{seed}.csv"}}
            for seed in SEEDS for model, modes in MODELS.items()]


def part_b_jobs():
    jobs = []
    for kind in ("drop", "keep"):
        for seed in SEEDS:
            for group in INPUT_GROUPS:
                dropped = [group] if kind == "drop" else [g for g in INPUT_GROUPS if g != group]
                jobs.append({"job_id": f"ablation_nohist_{kind}-{group}_seed{seed}",
                             "overrides": {**FIXED, **MODELS["nohist"], "SEED": seed, "DROP_INPUT_GROUPS": dropped},
                             "log": {"ablation": f"{kind} {group}", "DROP_INPUT_GROUPS": "+".join(dropped)}})
    return jobs


PARTS = {
    "part_a": {"dir": SWEEP_DIR / "predictions", "log": "tier0_predictions_log.csv", "prefix": "P",
               "jobs": part_a_jobs, "extra": ["predictions_file"]},
    "part_b": {"dir": SWEEP_DIR / "ablation", "log": "tier0_ablation_log.csv", "prefix": "A",
               "jobs": part_b_jobs, "extra": ["ablation", "DROP_INPUT_GROUPS"]},
}


def setup(part):
    spec = PARTS[part]
    p = {k: spec["dir"] / k for k in ("results", "logs", "overrides")}
    for folder in p.values():
        folder.mkdir(parents=True, exist_ok=True)
    i = CONTROLS_COLUMNS.index("CURRENT_MODE") + 1            # the controls log columns, plus this part's own
    columns = CONTROLS_COLUMNS[:i] + spec["extra"] + CONTROLS_COLUMNS[i:]
    return p | {"log_csv": spec["dir"] / spec["log"], "stop": spec["dir"] / "STOP"}, columns, spec["prefix"]


def is_done(job, p):
    return (p["results"] / f"{job['job_id']}.json").exists()


def run_job(job, p, columns, prefix):
    """One job in its own process, as tier0_sweep.run_job does. Raises KeyboardInterrupt after logging it."""
    job_id = job["job_id"]
    override_path, result_path, log_path = (p["overrides"] / f"{job_id}.json", p["results"] / f"{job_id}.json",
                                            p["logs"] / f"{job_id}.log")
    override_path.write_text(json.dumps(job["overrides"], indent=2), encoding="utf-8")
    env = {**os.environ, "TIER0_OVERRIDE_JSON": str(override_path), "RESULTS_PATH": str(result_path), "PYTHONUNBUFFERED": "1"}
    previous = list(csv.DictReader(open(p["log_csv"], newline="", encoding="utf-8"))) if p["log_csv"].exists() else []
    row = {"run": f"{prefix}{len(previous) + 1}", "job_id": job_id, **{k: job["overrides"][k] for k in SETTINGS},
           "HISTORY_MODE": job["overrides"]["HISTORY_MODE"], "CURRENT_MODE": job["overrides"]["CURRENT_MODE"], **job["log"]}
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
    new = not p["log_csv"].exists()
    with open(p["log_csv"], "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})
    if interrupted:
        raise KeyboardInterrupt
    return row


def main():
    parser = argparse.ArgumentParser(description="Run or resume a Tier 0 follow-up part.")
    parser.add_argument("part", choices=sorted(PARTS))
    parser.add_argument("--status", action="store_true", help="print progress only; no training")
    args = parser.parse_args()
    jobs = PARTS[args.part]["jobs"]()
    p, columns, prefix = setup(args.part)
    remaining = [j for j in jobs if not is_done(j, p)]
    print(f"{args.part}: {len(jobs) - len(remaining)} of {len(jobs)} jobs done, {len(remaining)} to run")
    if args.status:
        for j in jobs:
            print(f"  {'done' if is_done(j, p) else '    '}  {j['job_id']}")
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
            row = run_job(job, p, columns, prefix)
            extra = (f"  balanced accuracy {float(row['balanced_accuracy_model']):.4f}" if row["status"] == "OK" else f": {row['notes']}")
            print(f"    {row['status']} in {(time.time() - started) / 60:.1f} min{extra}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. The running job is logged as CRASHED (KeyboardInterrupt) and runs again on resume.")
        return
    left = sum(not is_done(j, p) for j in jobs)
    print(f"{args.part} finished: {len(jobs) - left} of {len(jobs)} done" + (f", {left} crashed (rerun this script)" if left else ""))


if __name__ == "__main__":
    main()
