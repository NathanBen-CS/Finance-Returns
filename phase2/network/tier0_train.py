"""
tier0_train.py -- STEP 2 of 2: build each example's history, train, and compare to a baseline.

An example = one event. The model sees:
  history   up to MAX_HISTORY_LEN earlier events of the same company (history_batch below)
  current   what this filing is + the company profile -- never this event's own reaction
and predicts this event's CAR[-1,+1] (and, as a side task, its reaction_category).

Split is by TIME (walk-forward): train on the past, validate on the next six months,
test on the most recent period. Results are printed next to a naive baseline
(always predict the training-set average), which is the honest yardstick.

Run:  python tier0_train.py
GPU:  an RTX 50-series card needs PyTorch >= 2.7 built for CUDA 12.8:
      pip install torch --index-url https://download.pytorch.org/whl/cu128
      Without it, get_device() falls back to the CPU and says why.
"""
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

import tier0_config as cfg
from tier0_model import Tier0Model

def history_columns(d):
    """Names of the columns in a history row, in order (handy for inspecting history_for output)."""
    cycles = [f"{f}(2pi*days_ago/{p})" for p in cfg.TIME_CYCLE_PERIODS for f in ("sin", "cos")]
    return (d["event_columns"]
            + ["log_trading_days_ago"] + (cycles if cfg.USE_TIME_CYCLES else [])
            + ["abnormal_return", "volume_change", "car1", "car5"]
            + [f"reaction={c}" for c in d["reaction_classes"]]
            + ["day0_known", "volume_known", "car1_known", "car5_known"])


TABLE_KEYS = ["event_x", "company_x", "abnormal_return", "volume_change", "car1", "car5",
              "volume_present", "reaction_onehot", "trading_day", "company_start",
              "target_car1", "target_class"]
LOOKAHEAD_COLUMNS = ("volatility_tier=", "log_trailing_volatility_1y")   # removed when DROP_LOOKAHEAD_FEATURES

# The groups DROP_INPUT_GROUPS can zero, as tests on the current-input column names (event columns
# first, then company columns). Every one of the 63 columns falls in exactly one group.
INPUT_GROUPS = {
    "category_items": lambda c: c.startswith(("event_category=", "item=")) or c == "num_items_disclosed",
    "timing":         lambda c: c.startswith("filing_timeliness=") or c in ("log_days_since_prev_filing", "is_first_event"),
    "size":           lambda c: c.startswith("size_category=") or c == "log_market_cap",
    "sector":         lambda c: c.startswith("sector="),
    "volatility":     lambda c: c.startswith(LOOKAHEAD_COLUMNS),
    "other_profile":  lambda c: c in ("log_revenue_ttm", "log_employees"),
}
PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep", "predictions")


def current_columns(d):
    return d["event_columns"] + d["company_columns"]


def group_columns(d):
    """{group: [its column names]}, checked to cover every current column exactly once."""
    columns = current_columns(d)
    groups = {g: [c for c in columns if test(c)] for g, test in INPUT_GROUPS.items()}
    counts = {c: sum(c in cols for cols in groups.values()) for c in columns}
    assert all(n == 1 for n in counts.values()), f"columns not in exactly one group: {[c for c, n in counts.items() if n != 1]}"
    return groups


def load_tables(d, device):
    """Copy every per-event array the batches need onto the device ONCE (~90 MB).
    After this, building a batch is a handful of GPU operations, not CPU work."""
    return {k: torch.as_tensor(d[k]).to(device) for k in TABLE_KEYS}


def history_batch(t, rows):
    """The histories of a whole batch of events at once, built where the tables live (the GPU).

    For each event: one row per earlier event of the same company (at most
    MAX_HISTORY_LEN, oldest first), padded with zero rows after the real ones. A past
    event's reaction is shown only if it was fully measured before this event's own
    window opened -- the rule and a worked example are in tier0_config.py. A hidden
    value is 0 with its *_known flag 0; a same-day filing has gap 0, so its reaction
    is always hidden.

    t: tables from load_tables; rows: (B,) row numbers of the events.
    Returns history (B, L, n_history_features) and history_len (B,)."""
    # how many past events each one gets: back to the company's first row, at most 128
    history_len = torch.clamp(rows - t["company_start"][rows], max=cfg.MAX_HISTORY_LEN)
    if cfg.HISTORY_MODE == "none":                                 # control: no earlier filings for anyone
        history_len = torch.zeros_like(history_len)
    L = int(history_len.max())                                   # longest history in this batch
    step = torch.arange(L, device=rows.device)
    real = step[None, :] < history_len[:, None]                    # (B, L): False = padding slot
    # (B, L) grid: the row number of each past event, oldest first. Padding slots point at
    # the event itself just to have a valid row -- they are zeroed at the end.
    past = torch.where(real, rows[:, None] - history_len[:, None] + step[None, :], rows[:, None])

    gap = t["trading_day"][rows][:, None] - t["trading_day"][past]  # (B, L) trading days ago
    day0_known = (gap >= cfg.MIN_GAP_DAY0) & real
    car1_known = (gap >= cfg.MIN_GAP_CAR1) & real
    car5_known = (gap >= cfg.MIN_GAP_CAR5) & real
    volume_known = day0_known & t["volume_present"][past]

    # Time maths in 64-bit: sin(2*pi*gap/p) with gaps of ~2,700 days loses precision in 32-bit.
    gap = gap.double()
    time_cols = [torch.log1p(gap)]                                 # how long ago
    if cfg.USE_TIME_CYCLES:                                        # where in each weekly/monthly/... cycle
        for p in cfg.TIME_CYCLE_PERIODS:
            time_cols += [torch.sin(2 * torch.pi * gap / p), torch.cos(2 * torch.pi * gap / p)]

    column = lambda x: x.float()[..., None]                        # (B, L) -> (B, L, 1) number column
    history = torch.cat([
        t["event_x"][past],                                        # what each past filing was
        *[column(c) for c in time_cols],
        column(t["abnormal_return"][past] * day0_known),           # its reaction, if knowable yet
        column(t["volume_change"][past] * volume_known),
        column(t["car1"][past] * car1_known),
        column(t["car5"][past] * car5_known),
        t["reaction_onehot"][past] * car1_known[..., None],
        column(day0_known), column(volume_known), column(car1_known), column(car5_known),
    ], dim=-1)
    return history * real[..., None], history_len                  # padding slots become all zeros


def history_for(d, i):
    """One event's history as a NumPy table (for inspecting). Same code path as training."""
    history, history_len = history_batch(load_tables(d, "cpu"), torch.tensor([i]))
    return history[0, :history_len[0]].numpy()


def batches(t, rows, shuffle):
    """Serve (history, history_len, current, target_car1, target_class) batches, all built on
    the device. Replaces a DataLoader: no CPU workers, no per-example Python."""
    if shuffle:                                                    # new random order every epoch
        rows = rows[torch.randperm(len(rows), device=rows.device)]
    for start in range(0, len(rows), cfg.BATCH_SIZE):
        r = rows[start:start + cfg.BATCH_SIZE]
        history, history_len = history_batch(t, r)
        current = torch.cat([t["event_x"][r], t["company_x"][r]], dim=-1)   # 40 + 23 = 63 numbers
        if cfg.CURRENT_MODE == "none":                             # control: the current inputs all zero
            current = torch.zeros_like(current)
        if "current_keep" in t:                                    # DROP_INPUT_GROUPS: zero the dropped columns
            current = current * t["current_keep"]
        yield history, history_len, current, t["target_car1"][r], t["target_class"][r]


def get_device():
    """is_available() can say True for a GPU this PyTorch build cannot run, so test a real op."""
    if torch.cuda.is_available():
        try:
            (torch.zeros(8, device="cuda") + 1).sum().item()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"GPU found but unusable with this PyTorch build ({str(e).splitlines()[0]}).\n"
                  "Using CPU. See the install note at the top of this file.")
    return torch.device("cpu")


def run_epoch(model, t, rows, pos_weight, optimizer=None):
    """One pass over the events in rows; trains if an optimizer is given. Returns the average loss."""
    training = optimizer is not None
    model.train(training)
    # regression: Huber loss (robust to fat-tailed CARs); direction: binary cross-entropy on a logit,
    # with up events weighted by pos_weight so neither direction dominates
    car_loss = nn.SmoothL1Loss() if cfg.TASK == "regression" else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    class_loss = nn.CrossEntropyLoss()
    total = 0.0
    with torch.set_grad_enabled(training):
        for history, lengths, current, car, cls in batches(t, rows, shuffle=training):
            car_pred, logits = model(history, lengths, current)
            target = car if cfg.TASK == "regression" else (car > 0).float()   # direction: 1 = up, 0 = down
            loss = car_loss(car_pred, target) + cfg.REACTION_LOSS_WEIGHT * class_loss(logits, cls)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(car)
    return total / len(rows)


@torch.no_grad()
def predict(model, t, rows):
    model.eval()
    car_pred, class_pred = [], []
    for history, lengths, current, _, _ in batches(t, rows, shuffle=False):
        c, logits = model(history, lengths, current)
        car_pred.append(c.cpu().numpy())
        class_pred.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(car_pred), np.concatenate(class_pred)


def save_predictions(d, test_rows, car_pred, class_pred, m):
    """One row per test event (the same rows the metrics use), then read back and re-scored."""
    import pandas as pd
    name = os.path.splitext(os.path.basename(os.environ["RESULTS_PATH"]))[0] if os.environ.get("RESULTS_PATH") else "manual_run"
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    path = os.path.join(PREDICTIONS_DIR, f"{name}.csv")
    classes = np.array(d["reaction_classes"])
    car_true = d["target_car1"][test_rows]
    direction = cfg.TASK == "direction"
    score = 1 / (1 + np.exp(-car_pred.astype(np.float64))) if direction else car_pred   # logit -> P(up)
    table = pd.DataFrame({
        "event_id": d["event_id"][test_rows], "ticker": d["ticker"][test_rows],
        "event_date": pd.to_datetime(d["event_date"][test_rows]).strftime("%Y-%m-%d"),   # report_date
        "trading_day": d["trading_day"][test_rows],                                      # day 0 on the market calendar
        "true_car1": car_true, "true_up": (car_true > 0).astype(int),
        ("pred_prob_up" if direction else "pred_car1"): score, "pred_up": (car_pred > 0).astype(int),
        "true_reaction": classes[d["target_class"][test_rows]], "pred_reaction": classes[class_pred],
    })
    table.to_csv(path + ".tmp", index=False, float_format="%.9g")
    os.replace(path + ".tmp", path)

    back = pd.read_csv(path, keep_default_na=False)               # keep_default_na: a ticker "NA" stays text
    up = back["true_up"] == 1
    recomputed = {"reaction_accuracy_model": np.mean(back["true_reaction"] == back["pred_reaction"])}
    if direction:
        recomputed["balanced_accuracy_model"] = (np.mean(back["pred_up"][up] == 1) + np.mean(back["pred_up"][~up] == 0)) / 2
    for key, value in recomputed.items():
        assert abs(value - m[key]) < 1e-12, f"{key} recomputed from {path} is {value!r}, the run logged {m[key]!r}"
    print(f"predictions written to {path} ({len(back):,} rows); recomputed from the file: "
          + ", ".join(f"{k} {v:.6f} (logged {m[k]:.6f})" for k, v in recomputed.items()))


def main():
    assert cfg.TASK in ("regression", "direction"), f"unknown TASK {cfg.TASK!r}"
    assert cfg.HISTORY_MODE in ("full", "none") and cfg.CURRENT_MODE in ("full", "none"), \
        f"HISTORY_MODE / CURRENT_MODE must be 'full' or 'none', got {cfg.HISTORY_MODE!r} / {cfg.CURRENT_MODE!r}"
    unknown = set(cfg.DROP_INPUT_GROUPS) - set(INPUT_GROUPS)
    assert not unknown, f"unknown DROP_INPUT_GROUPS {sorted(unknown)}; the groups are {list(INPUT_GROUPS)}"
    started = time.time()
    torch.manual_seed(cfg.SEED)          # same starting weights and shuffle order every run
    np.random.seed(cfg.SEED)
    with open(cfg.EVENTS_PKL, "rb") as f:
        d = pickle.load(f)
    if cfg.DROP_LOOKAHEAD_FEATURES:      # done here too, so it works whether or not prepare already dropped them
        keep = [i for i, c in enumerate(d["company_columns"]) if not c.startswith(LOOKAHEAD_COLUMNS)]
        d["company_x"] = d["company_x"][:, keep]
        d["company_columns"] = [d["company_columns"][i] for i in keep]
    print(f"Settings: USE_TIME_CYCLES={cfg.USE_TIME_CYCLES}, DROP_LOOKAHEAD_FEATURES={cfg.DROP_LOOKAHEAD_FEATURES}, "
          f"CAUSAL_READOUT={cfg.CAUSAL_READOUT}, TASK={cfg.TASK}, CLASS_WEIGHTED_LOSS={cfg.CLASS_WEIGHTED_LOSS}, "
          f"SEED={cfg.SEED}, HISTORY_MODE={cfg.HISTORY_MODE}, CURRENT_MODE={cfg.CURRENT_MODE}, "
          f"DROP_INPUT_GROUPS={list(cfg.DROP_INPUT_GROUPS)}, SAVE_PREDICTIONS={cfg.SAVE_PREDICTIONS}")

    dates = d["event_date"]
    train_rows = np.flatnonzero(dates < np.datetime64(cfg.TRAIN_END))
    val_rows = np.flatnonzero((dates >= np.datetime64(cfg.TRAIN_END)) & (dates < np.datetime64(cfg.VAL_END)))
    test_rows = np.flatnonzero(dates >= np.datetime64(cfg.VAL_END))
    print(f"Split: {len(train_rows):,} train / {len(val_rows):,} validation / {len(test_rows):,} test")

    device = get_device()
    t = load_tables(d, device)                                     # all event data onto the GPU, once
    if cfg.DROP_INPUT_GROUPS:
        dropped = {c for g in cfg.DROP_INPUT_GROUPS for c in group_columns(d)[g]}
        keep = [c not in dropped for c in current_columns(d)]
        t["current_keep"] = torch.tensor(keep, dtype=torch.float32, device=device)
        print(f"DROP_INPUT_GROUPS: zeroing {len(dropped)} of {len(keep)} current inputs: {sorted(dropped)}")
    on_device = lambda rows: torch.as_tensor(rows, device=device)  # row numbers as a GPU tensor
    n_history = len(history_columns(d))
    n_current = len(d["event_columns"]) + len(d["company_columns"])
    model = Tier0Model(n_history, n_current, n_classes=len(d["reaction_classes"]),
                       causal_readout=cfg.CAUSAL_READOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    # direction task: weight each up event by (down events / up events), counted on the TRAINING
    # split only and fixed before training starts (ignored when TASK = "regression")
    train_car = d["target_car1"][train_rows]
    pos_weight = (torch.tensor((train_car <= 0).sum() / (train_car > 0).sum(), device=device)
                  if cfg.CLASS_WEIGHTED_LOSS else None)

    epochs_completed = 0
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = run_epoch(model, t, on_device(train_rows), pos_weight, optimizer)
        val_loss = run_epoch(model, t, on_device(val_rows), pos_weight)
        epochs_completed = epoch
        print(f"epoch {epoch:2d}/{cfg.EPOCHS}  train loss {train_loss:.4f}  validation loss {val_loss:.4f}")

    # ---------------- is it better than guessing? (test set = the most recent, never-seen period)
    car_pred, class_pred = predict(model, t, on_device(test_rows))
    car_true, class_true = d["target_car1"][test_rows], d["target_class"][test_rows]
    baseline = d["target_car1"][train_rows].mean()                 # "always predict the average"
    majority = np.bincount(d["target_class"][train_rows]).argmax()  # "always predict the commonest class"
    majority_direction = 1 if (train_car > 0).sum() >= (train_car < 0).sum() else -1  # "always predict the commoner direction"

    m = {}   # every number printed below, under the names of tier0_run_log.csv
    print(f"\nTEST RESULTS (model vs. naive baseline), TASK={cfg.TASK}")
    if cfg.TASK == "regression":
        def r2_vs_baseline(mask):
            return 1 - ((car_true[mask] - car_pred[mask]) ** 2).sum() / ((car_true[mask] - baseline) ** 2).sum()

        everything = np.ones(len(car_true), bool)
        ordinary = np.abs(car_true) <= 0.5  # the >50% tail includes data glitches and dominates squared error
        m["CAR_MAE_model"], m["CAR_MAE_baseline"] = np.abs(car_true - car_pred).mean(), np.abs(car_true - baseline).mean()
        m["R2_all_events"], m["R2_trimmed"] = r2_vs_baseline(everything), r2_vs_baseline(ordinary)
        print(f"  CAR MAE              model {m['CAR_MAE_model']:.5f}   baseline {m['CAR_MAE_baseline']:.5f}")
        print(f"  R^2 vs baseline      all events {m['R2_all_events']:+.4f}   "
              f"|CAR| <= 50% ({ordinary.mean():.1%} of events) {m['R2_trimmed']:+.4f}")
        # direction only: did the predicted CAR land on the same side of zero as the real one?
        # (the baseline always guesses whichever direction was commoner in training)
        m["direction_accuracy_model"] = np.mean(np.sign(car_pred) == np.sign(car_true))
        m["direction_accuracy_baseline"] = np.mean(majority_direction == np.sign(car_true))
        print(f"  direction accuracy   model {m['direction_accuracy_model']:.3f}   baseline {m['direction_accuracy_baseline']:.3f}")
    else:
        # car_pred holds logits: above 0 means "predicted up" (probability above 0.5)
        up_true, up_pred = car_true > 0, car_pred > 0
        m["direction_accuracy_model"] = np.mean(up_pred == up_true)
        m["direction_accuracy_baseline"] = np.mean((majority_direction > 0) == up_true)
        print(f"  direction accuracy   model {m['direction_accuracy_model']:.3f}   baseline {m['direction_accuracy_baseline']:.3f}")
        # recall per side: a model that always guesses one direction scores 1.000 on one, 0.000 on the other
        recall_up, recall_down = np.mean(up_pred[up_true]), np.mean(~up_pred[~up_true])
        m["recall_up_events"], m["recall_down_events"] = recall_up, recall_down
        print(f"  recall               up events {recall_up:.3f}   down events {recall_down:.3f}")
        # balanced accuracy: the two recalls averaged, so both directions count equally.
        # Always guessing one side gives (1 + 0) / 2 = 0.500 whatever the class balance.
        m["balanced_accuracy_model"], m["balanced_accuracy_baseline"] = (recall_up + recall_down) / 2, 0.5
        print(f"  balanced accuracy    model {m['balanced_accuracy_model']:.3f}   baseline 0.500")
    m["reaction_accuracy_model"], m["reaction_accuracy_baseline"] = np.mean(class_pred == class_true), np.mean(majority == class_true)
    print(f"  reaction accuracy    model {m['reaction_accuracy_model']:.3f}   baseline {m['reaction_accuracy_baseline']:.3f}")
    print("Above 0 / above baseline = real signal. At or below = this feature set does not predict CAR.")
    if cfg.SAVE_PREDICTIONS:
        save_predictions(d, test_rows, car_pred, class_pred, m)

    if os.environ.get("RESULTS_PATH"):   # set by tier0_sweep.py; a plain run writes nothing
        settings = {k: v for k, v in vars(cfg).items() if k.isupper() and isinstance(v, (bool, int, float, str, tuple))}
        results = {
            "settings": {k: settings[k] for k in ("TASK", "USE_TIME_CYCLES", "CAUSAL_READOUT", "DROP_LOOKAHEAD_FEATURES", "SEED")}
                        | {"CLASS_WEIGHTED_LOSS": cfg.CLASS_WEIGHTED_LOSS if cfg.TASK == "direction" else "N/A",
                           "HISTORY_MODE": cfg.HISTORY_MODE, "CURRENT_MODE": cfg.CURRENT_MODE,
                           "DROP_INPUT_GROUPS": list(cfg.DROP_INPUT_GROUPS)},
            "metrics": {k: float(v) for k, v in m.items()},
            "runtime_s": time.time() - started, "epochs_completed": epochs_completed,
            "events": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows),
                       "dropped_unreliable": d.get("dropped_unreliable")},
            "all_settings": settings,
        }
        path = os.environ["RESULTS_PATH"]
        with open(path + ".tmp", "w", encoding="utf-8") as f:   # written whole, then renamed, so a crash never leaves half a file
            json.dump(results, f, indent=2)
        os.replace(path + ".tmp", path)
        print(f"results written to {path}")


if __name__ == "__main__":
    main()
