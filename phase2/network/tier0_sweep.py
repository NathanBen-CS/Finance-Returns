"""
tier0_sweep.py -- run every Tier 0 setting combination unattended, and pick up where it left off.

The grid: TASK x USE_TIME_CYCLES x CAUSAL_READOUT x DROP_LOOKAHEAD_FEATURES, plus CLASS_WEIGHTED_LOSS
when TASK is "direction" = 24 settings, each with SEED 0, 1, 2 = 72 jobs, always in the same order.
Each job runs tier0_train.py as a separate process with its settings in a JSON file
(TIER0_OVERRIDE_JSON) and writes its results to sweep/results/<job_id>.json (RESULTS_PATH).

Run / resume:  python tier0_sweep.py            (skips every job that already has a results file)
Progress:      python tier0_sweep.py --status   (no training)
Stop:          create the file sweep/STOP. The sweep stops before starting its next job and
               deletes STOP, so the next run resumes. Ctrl+C stops at once; the job it was
               running is logged as CRASHED (KeyboardInterrupt) and runs again on resume.

Writes, in sweep/:  results/<job_id>.json   one per finished job (the done marker)
                    tier0_sweep_log.csv     one row per attempt, finished or crashed
                    logs/<job_id>.log       the full output of the job's last attempt
                    overrides/<job_id>.json the settings handed to the job

The pickle must be built first, once:  python tier0_prepare_data.py  (with DROP_LOOKAHEAD_FEATURES =
False, its default: tier0_train.py drops those columns itself when a job asks for it).
"""
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import tier0_config as cfg

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / "sweep"
TRAIN_SCRIPT = HERE / "tier0_train.py"
SEEDS = [0, 1, 2]
FIRST_RUN = 27              # tier0_run_log.csv, kept by hand outside this repo, ends at run 26
JOB_TIMEOUT_HOURS = 3       # one job takes about 20 minutes on an RTX 5060 Ti; a hung job is killed after this
FIXED_OVERRIDES = {}        # added to every job; empty for the real sweep (a quick test can set {"EPOCHS": 1})

SETTINGS = ["TASK", "USE_TIME_CYCLES", "CAUSAL_READOUT", "DROP_LOOKAHEAD_FEATURES", "CLASS_WEIGHTED_LOSS", "SEED"]
METRICS = ["CAR_MAE_model", "CAR_MAE_baseline", "R2_all_events", "R2_trimmed",
           "reaction_accuracy_model", "reaction_accuracy_baseline", "direction_accuracy_model", "direction_accuracy_baseline",
           "recall_up_events", "recall_down_events", "balanced_accuracy_model", "balanced_accuracy_baseline"]
LOG_COLUMNS = ["run", "job_id", "finished_at", "status", *SETTINGS, *METRICS, "runtime_s", "epochs_completed", "notes"]


def build_jobs():
    """All 72 jobs in a fixed order. Seed 0 of every setting comes first, then seed 1, then seed 2,
    so a sweep stopped partway has already covered the whole grid once."""
    jobs = []
    for seed in SEEDS:
        for task in ("regression", "direction"):
            for use_time_cycles, causal_readout, drop_lookahead in product((False, True), repeat=3):
                for class_weighted in ((None,) if task == "regression" else (False, True)):
                    cw = "na" if class_weighted is None else int(class_weighted)
                    overrides = {"TASK": task, "USE_TIME_CYCLES": use_time_cycles, "CAUSAL_READOUT": causal_readout,
                                 "DROP_LOOKAHEAD_FEATURES": drop_lookahead, "SEED": seed}
                    if class_weighted is not None:
                        overrides["CLASS_WEIGHTED_LOSS"] = class_weighted
                    job_id = f"{task}_tc{int(use_time_cycles)}_cr{int(causal_readout)}_drop{int(drop_lookahead)}_cw{cw}_seed{seed}"
                    jobs.append({"job_id": job_id, "overrides": overrides})
    assert len(jobs) == 72 and len({j["job_id"] for j in jobs}) == 72
    return jobs


def paths(sweep_dir):
    p = {"results": sweep_dir / "results", "logs": sweep_dir / "logs", "overrides": sweep_dir / "overrides",
         "log_csv": sweep_dir / "tier0_sweep_log.csv", "stop": sweep_dir / "STOP"}
    for key in ("results", "logs", "overrides"):
        p[key].mkdir(parents=True, exist_ok=True)
    return p


def read_log(p):
    if not p["log_csv"].exists():
        return []
    with open(p["log_csv"], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(p, row):
    new = not p["log_csv"].exists()
    with open(p["log_csv"], "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def progress(jobs, p):
    done = [j for j in jobs if (p["results"] / f"{j['job_id']}.json").exists()]
    crashed_ids = {r["job_id"] for r in read_log(p) if r["status"].startswith("CRASHED")}
    crashed = [j for j in jobs if j["job_id"] in crashed_ids and j not in done]
    remaining = [j for j in jobs if j not in done]
    return done, crashed, remaining


def hours(seconds):
    return f"{seconds / 3600:.1f} h ({seconds / 86400:.1f} days)" if seconds >= 86400 else f"{seconds / 3600:.1f} h"


def running_job(jobs, p):
    """The job whose log is still open: no results file yet and no stderr section, which run_job adds when the job ends."""
    by_id = {j["job_id"]: j for j in jobs}
    for log in sorted(p["logs"].glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True):
        job = by_id.get(log.stem)
        text = log.read_text(encoding="utf-8", errors="replace")
        if job and not (p["results"] / f"{log.stem}.json").exists() and "---------------- stderr" not in text:
            return job, log, text
    return None


def status(jobs, p):
    done, crashed, remaining = progress(jobs, p)
    print(f"{len(done)} of {len(jobs)} jobs done, {len(crashed)} crashed on their last attempt (they run again on resume), {len(remaining)} remaining")
    running = running_job(jobs, p)
    if running:
        job, log, text = running
        epochs = [line for line in text.splitlines() if line.startswith("epoch")]
        quiet = (time.time() - log.stat().st_mtime) / 60
        print(f"running now: job #{jobs.index(job) + 1} {job['job_id']}, started {datetime.fromtimestamp(log.stat().st_ctime):%H:%M}"
              + (f", {epochs[-1].strip()}" if epochs else ", loading data"))
        if quiet > 10:
            print(f"    but its log has not changed for {quiet:.0f} min: the sweep may have been closed (resume it with python tier0_sweep.py)")
    elif remaining:
        print(f"next: job #{jobs.index(remaining[0]) + 1} {remaining[0]['job_id']}")
    runtimes = [json.loads((p["results"] / f"{j['job_id']}.json").read_text(encoding="utf-8"))["runtime_s"] for j in done]
    if runtimes and remaining:
        print(f"at {sum(runtimes) / len(runtimes) / 60:.0f} min per job, the rest takes about {hours(sum(runtimes) / len(runtimes) * len(remaining))}")
    if p["stop"].exists():
        print("STOP file present: the next run stops before its first job.")


def check_pickle():
    """Refuse to start on a pickle built before the price filter, or without the look-ahead columns."""
    with open(cfg.EVENTS_PKL, "rb") as f:
        d = pickle.load(f)
    assert "dropped_unreliable" in d, "tier0_events.pkl predates the price_data_unreliable filter: run tier0_prepare_data.py"
    assert any(c.startswith("volatility_tier=") for c in d["company_columns"]), \
        "tier0_events.pkl was built with DROP_LOOKAHEAD_FEATURES = True: rebuild it with False so both sides of the grid can run"
    print(f"pickle OK: {len(d['event_id']):,} events, {d['dropped_unreliable']} unreliable events dropped")


def stderr_tail(text, lines=4, limit=600):
    tail = " | ".join(line.strip() for line in text.strip().splitlines()[-lines:] if line.strip())
    return tail[-limit:]


def run_job(job, number, p):
    """Run one job in its own process. Returns its log row; raises KeyboardInterrupt after logging one."""
    job_id = job["job_id"]
    override_path, result_path, log_path = p["overrides"] / f"{job_id}.json", p["results"] / f"{job_id}.json", p["logs"] / f"{job_id}.log"
    override_path.write_text(json.dumps({**job["overrides"], **FIXED_OVERRIDES}, indent=2), encoding="utf-8")
    env = {**os.environ, "TIER0_OVERRIDE_JSON": str(override_path), "RESULTS_PATH": str(result_path), "PYTHONUNBUFFERED": "1"}
    settings = {**job["overrides"], "CLASS_WEIGHTED_LOSS": job["overrides"].get("CLASS_WEIGHTED_LOSS", "N/A")}
    row = {"run": number, "job_id": job_id, **settings}
    interrupted = False
    with open(log_path, "w", encoding="utf-8") as out:
        process = subprocess.Popen([sys.executable, str(TRAIN_SCRIPT)], cwd=HERE, env=env, stdout=out, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace")
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
        row.update(status="OK", **result["metrics"], runtime_s=round(result["runtime_s"], 1), epochs_completed=result["epochs_completed"],
                   notes=f"settings beyond the grid: {FIXED_OVERRIDES}" if FIXED_OVERRIDES else "")
    append_row(p, row)
    if interrupted:
        raise KeyboardInterrupt
    return row


def sweep(jobs, p):
    check_pickle()
    done, _, remaining = progress(jobs, p)
    print(f"{len(done)} of {len(jobs)} jobs already done, {len(remaining)} to run. Stop between jobs by creating {p['stop']}")
    durations = []
    for job in remaining:
        if p["stop"].exists():
            p["stop"].unlink()
            print("STOP file found: stopping before the next job (STOP removed, so the next run resumes).")
            return
        if (p["results"] / f"{job['job_id']}.json").exists():   # finished by another copy of the sweep meanwhile
            continue
        number = max([int(r["run"]) for r in read_log(p)], default=FIRST_RUN - 1) + 1
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] run {number}: job #{jobs.index(job) + 1}/{len(jobs)} {job['job_id']} ...", flush=True)
        started = time.time()
        row = run_job(job, number, p)
        took = time.time() - started
        print(f"    {row['status']} in {took / 60:.1f} min" + (f": {row['notes']}" if row["status"] != "OK" else ""), flush=True)
        if row["status"] != "OK":
            continue
        durations.append(took)                                  # only finished jobs count toward the estimate
        left = len(progress(jobs, p)[2])
        if len(durations) <= 2 and left:
            print(f"    estimate: {left} jobs left at about {sum(durations) / len(durations) / 60:.0f} min each = {hours(sum(durations) / len(durations) * left)}", flush=True)
    done, crashed, remaining = progress(jobs, p)
    print(f"sweep pass finished: {len(done)} done, {len(crashed)} crashed (rerun them with another python tier0_sweep.py)")


def main(argv=None, sweep_dir=SWEEP_DIR):
    parser = argparse.ArgumentParser(description="Run or resume the Tier 0 settings sweep.")
    parser.add_argument("--status", action="store_true", help="print progress only; no training")
    args = parser.parse_args(argv)
    jobs, p = build_jobs(), paths(sweep_dir)
    if args.status:
        status(jobs, p)
    else:
        try:
            sweep(jobs, p)
        except KeyboardInterrupt:
            print("\nInterrupted. A job that was running is logged as CRASHED (KeyboardInterrupt) and runs again on resume.")


if __name__ == "__main__":
    main()
