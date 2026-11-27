"""
bootstrap_ci.py -- 95% intervals for balanced accuracy from the saved test predictions (read-only on them).

Test events are not independent: one company files many times, and many filings share a trading day
(the same market move). So besides the naive event-level bootstrap, whole clusters are resampled:

  company  every test event of a resampled company comes along (cluster = ticker)
  day      every test event of a resampled trading day comes along (cluster = trading day of day 0,
           the day the CAR window is anchored on; report_dates that fall on a weekend share Monday's cluster)
  event    each event on its own (the naive bootstrap, for comparison)

2,000 resamples each, percentile intervals. Within one cluster type every run uses the SAME resamples,
so the paired differences (full - no-history, full - history-only, per seed) resample the same clusters
for both models.

Reads:   sweep/predictions/pred_*.csv, sweep/predictions/tier0_predictions_log.csv,
         sweep/tier0_sweep_log.csv and sweep/controls/tier0_controls_log.csv (the original runs, for the rerun comparison)
Writes:  sweep/predictions/bootstrap_ci.csv, bootstrap_paired.csv, bootstrap_summary.txt

Run:  python bootstrap_ci.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRED = HERE / "sweep" / "predictions"
RESAMPLES = 2000
SEED = 2026
CHUNK = 100
CLUSTERS = {"company": "ticker", "day": "trading_day", "event": None}
MODELS = ("full", "nohist", "histonly")
# the original runs these are reruns of (same settings and seed)
ORIGINALS = {"pred_full_seed0": ("sweep", "48"), "pred_full_seed1": ("sweep", "72"),
             "pred_nohist_seed0": ("controls", "C3"), "pred_nohist_seed1": ("controls", "C6"),
             "pred_histonly_seed0": ("controls", "C5"), "pred_histonly_seed1": ("controls", "C8")}


def balanced(up_hit, up_n, down_hit, down_n):
    return (up_hit / up_n + down_hit / down_n) / 2


def load_runs():
    runs = {}
    for path in sorted(PRED.glob("pred_*.csv")):
        runs[path.stem] = pd.read_csv(path, keep_default_na=False)   # keep_default_na: a ticker "NA" stays text
    assert runs, f"no prediction files in {PRED}"
    first = next(iter(runs.values()))
    for name, table in runs.items():
        assert table["event_id"].equals(first["event_id"]), f"{name} has different test events (or order) than the others"
        assert table["true_up"].equals(first["true_up"]), f"{name} has different true labels than the others"
    return runs, first


def bootstrap(runs, truth, key, rng):
    """(RESAMPLES, n_runs) balanced accuracies, every run on the same cluster resamples."""
    codes = np.arange(len(truth)) if key is None else pd.factorize(truth[key])[0]
    k = codes.max() + 1
    up = truth["true_up"].to_numpy() == 1
    per_cluster = lambda mask: np.bincount(codes, weights=mask.astype(float), minlength=k)
    up_n, down_n = per_cluster(up), per_cluster(~up)
    # (k, n_runs) per-cluster counts of correctly called up events and correctly called down events
    hits_up = np.stack([per_cluster(up & (t["pred_up"].to_numpy() == 1)) for t in runs.values()], axis=1)
    hits_down = np.stack([per_cluster(~up & (t["pred_up"].to_numpy() == 0)) for t in runs.values()], axis=1)
    out = []
    for start in range(0, RESAMPLES, CHUNK):
        n = min(CHUNK, RESAMPLES - start)
        # row b = how many times each cluster was drawn in resample b (k draws with replacement)
        weights = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n)]).astype(float)
        out.append(balanced(weights @ hits_up, (weights @ up_n)[:, None], weights @ hits_down, (weights @ down_n)[:, None]))
    return np.concatenate(out), k


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return lo, hi


def original_scores():
    scores = {}
    for source, path in (("sweep", HERE / "sweep" / "tier0_sweep_log.csv"), ("controls", HERE / "sweep" / "controls" / "tier0_controls_log.csv")):
        for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
            if row["status"] == "OK" and row["balanced_accuracy_model"]:   # regression rows have none
                scores[(source, row["run"])] = (row["job_id"], float(row["balanced_accuracy_model"]))
    return scores


def main():
    runs, truth = load_runs()
    names = list(runs)
    logged = {r["job_id"]: float(r["balanced_accuracy_model"])
              for r in csv.DictReader(open(PRED / "tier0_predictions_log.csv", newline="", encoding="utf-8")) if r["status"] == "OK"}
    up = truth["true_up"].to_numpy() == 1
    point = {}
    for name, t in runs.items():
        pred = t["pred_up"].to_numpy() == 1
        point[name] = balanced(np.sum(pred & up), up.sum(), np.sum(~pred & ~up), (~up).sum())
        assert abs(point[name] - logged[name]) < 1e-12, f"{name}: recomputed {point[name]!r}, logged {logged[name]!r}"

    rng = np.random.default_rng(SEED)
    draws, n_clusters = {}, {}
    for cluster, key in CLUSTERS.items():
        draws[cluster], n_clusters[cluster] = bootstrap(runs, truth, key, rng)

    rows = []
    for j, name in enumerate(names):
        for cluster in CLUSTERS:
            lo, hi = interval(draws[cluster][:, j])
            rows.append({"run": name, "cluster": cluster, "n_clusters": n_clusters[cluster], "balanced_accuracy": point[name],
                         "ci_low": lo, "ci_high": hi, "ci_width": hi - lo,
                         "minus_half": point[name] - 0.5, "minus_half_ci_low": lo - 0.5, "minus_half_ci_high": hi - 0.5,
                         "excludes_50": bool(lo > 0.5 or hi < 0.5)})
    ci = pd.DataFrame(rows)
    ci.to_csv(PRED / "bootstrap_ci.csv", index=False)

    paired = []
    for seed in (0, 1):
        for other in ("nohist", "histonly"):
            a, b = f"pred_full_seed{seed}", f"pred_{other}_seed{seed}"
            if a not in runs or b not in runs:
                continue
            for cluster in CLUSTERS:
                diff = draws[cluster][:, names.index(a)] - draws[cluster][:, names.index(b)]
                lo, hi = interval(diff)
                paired.append({"seed": seed, "comparison": f"full - {other}", "cluster": cluster,
                               "difference": point[a] - point[b], "ci_low": lo, "ci_high": hi, "ci_width": hi - lo,
                               "excludes_0": bool(lo > 0 or hi < 0)})
    paired = pd.DataFrame(paired)
    paired.to_csv(PRED / "bootstrap_paired.csv", index=False)

    # ---------------------------------------------------------------- text summary
    originals = original_scores()
    pct = lambda x: f"{100 * x:.2f}"
    lines = [f"Cluster bootstrap of balanced accuracy: {RESAMPLES:,} resamples, percentile 95% intervals, seed {SEED}.",
             f"Test events: {len(truth):,}; clusters: {n_clusters['company']:,} companies, {n_clusters['day']:,} trading days.",
             "Recomputed balanced accuracy equals the logged value for every file (checked to 1e-12).", "",
             "RERUNS vs ORIGINAL RUNS (same settings and seed; GPU training is not bit-for-bit deterministic)"]
    for name in names:
        source, run = ORIGINALS.get(name, (None, None))
        job, original = originals.get((source, run), ("?", float("nan")))
        lines.append(f"  {name:22} rerun {pct(point[name])}%   original {source} run {run} ({job}) {pct(original)}%   "
                     f"difference {100 * (point[name] - original):+.2f} points")
    lines += ["", "BALANCED ACCURACY, 95% intervals (points; 'excl 50' = interval excludes 50%)"]
    for name in names:
        part = ci[ci.run == name].set_index("cluster")
        widest = part["ci_width"].idxmax()
        cells = "   ".join(f"{c}: [{pct(part.loc[c, 'ci_low'])}, {pct(part.loc[c, 'ci_high'])}]{' excl 50' if part.loc[c, 'excludes_50'] else ''}"
                           for c in CLUSTERS)
        lines.append(f"  {name:22} {pct(point[name])}%   {cells}   widest: {widest}")
    lines += ["", "PAIRED DIFFERENCES, same clusters resampled for both models (points; 'excl 0' = distinguishable from zero)"]
    for _, r in paired.iterrows():
        lines.append(f"  seed {r.seed}  {r.comparison:17} {r.cluster:8} {100 * r.difference:+.2f}   "
                     f"[{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]{'  excl 0' if r.excludes_0 else ''}")
    text = "\n".join(lines)
    (PRED / "bootstrap_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {PRED / 'bootstrap_ci.csv'}, {PRED / 'bootstrap_paired.csv'}, {PRED / 'bootstrap_summary.txt'}")


if __name__ == "__main__":
    main()
