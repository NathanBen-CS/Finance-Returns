"""
tier0_prepare_data.py -- STEP 1 of 2: turn the three CSVs into one table of events.

Output: tier0_events.pkl, a dict of NumPy arrays with ONE ROW PER EVENT, sorted by
company and date. Each row carries three groups of columns:

  event_x    what was filed           -- known the moment the filing exists
  outcome    how the market reacted   -- only shown to the model for PAST events,
                                         and only once their window has closed
  company_x  the company's profile    -- static (one Oct-2026 snapshot)

No histories are built here. tier0_train.py builds each example's history on the
fly from these rows (see history_for() there), which keeps this file small
(~130 MB instead of several GB) and keeps all of the leakage logic in one place.

Run:  python tier0_prepare_data.py
"""
import pickle

import numpy as np
import pandas as pd

import tier0_config as cfg

REACTION_CLASSES = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]


def signed_log(x):
    """sign(x) * log(1 + |x|): keeps the sign, squashes fat tails. Used for inputs only."""
    return np.sign(x) * np.log1p(np.abs(x))


def one_hot(series, prefix):
    """Category column -> 0/1 columns named 'prefix=value' (a missing value is all zeros)."""
    return pd.get_dummies(series, prefix=prefix, prefix_sep="=").astype(np.float32)


def main():
    # ---------------------------------------------------------------- 1. load + join
    read = dict(keep_default_na=False, na_values=[""])   # a real ticker "NA" must stay text
    companies = pd.read_csv(cfg.COMPANIES_CSV, **read)
    events = pd.read_csv(cfg.EVENTS_CSV, parse_dates=["filing_date", "report_date"], **read)
    reactions = pd.read_csv(cfg.MARKET_REACTIONS_CSV, parse_dates=["trading_date"], **read)

    df = events.merge(reactions, on="event_id", validate="one_to_one")
    df = df.merge(companies, on="ticker", how="left", validate="many_to_one")
    # Events whose price window has a split-like splice or a non-positive close (set by
    # ../data_phase2/flag_price_splices.py) are dropped first, so they are neither
    # examples nor part of anyone's history.
    assert df["price_data_unreliable"].dtype == bool
    unreliable = int(df["price_data_unreliable"].sum())
    df = df[~df["price_data_unreliable"]]
    # The Phase 2 pipeline drops every event without a complete price window, so
    # the reaction values are never missing. Check it rather than assume it.
    assert df[["abnormal_return", "car_minus1_plus1", "car_minus1_plus5"]].notna().all().all()

    # ---------------------------------------------------------------- 2. order
    # Each company's events in time order; event_id breaks same-day ties the same way every run.
    df = df.sort_values(["ticker", "report_date", "event_id"]).reset_index(drop=True)
    first_row = df.groupby("ticker").cumcount().to_numpy() == 0
    company_start = np.arange(len(df)) - df.groupby("ticker").cumcount().to_numpy()

    # Trading-day number of each event's day 0 (t0), counted on the market calendar
    # (every distinct trading date in the data). Gaps between events are then plain
    # subtraction, and weekends/holidays never count as time passing.
    calendar = np.sort(df["trading_date"].unique())
    trading_day = np.searchsorted(calendar, df["trading_date"].to_numpy())

    # ---------------------------------------------------------------- 3. event_x: what was filed
    days_since_prev = df.groupby("ticker")["report_date"].diff().dt.days.fillna(0).to_numpy()
    event_x = pd.concat([
        one_hot(df["event_category"], "event_category"),
        one_hot(df["filing_timeliness"], "filing_timeliness"),
        df["item_codes"].fillna("").str.get_dummies(sep=",").add_prefix("item=").astype(np.float32),
        pd.DataFrame({
            "num_items_disclosed": df["num_items_disclosed"].astype(np.float32),
            "log_days_since_prev_filing": np.log1p(days_since_prev).astype(np.float32),
            "is_first_event": first_row.astype(np.float32),
        }),
    ], axis=1)

    # ---------------------------------------------------------------- 4. company_x: the profile
    def log_or_zero(col):
        return np.log1p(df[col].fillna(0).clip(lower=0)).astype(np.float32)

    profile = [
        one_hot(df["sector"], "sector"),
        one_hot(df["size_category"], "size_category"),
        pd.DataFrame({"log_market_cap": log_or_zero("market_cap"),
                      "log_revenue_ttm": log_or_zero("revenue_ttm"),
                      "log_employees": log_or_zero("employees")}),
    ]
    if not cfg.DROP_LOOKAHEAD_FEATURES:
        profile += [one_hot(df["volatility_tier"], "volatility_tier"),
                    pd.DataFrame({"log_trailing_volatility_1y": log_or_zero("trailing_volatility_1y")})]
    company_x = pd.concat(profile, axis=1)

    # ---------------------------------------------------------------- 5. outcome: how the market reacted
    clip = cfg.CLIP_RETURNS
    volume = df["volume_change_pct"]
    outcome = {
        "abnormal_return": signed_log(df["abnormal_return"].clip(-clip, clip)).to_numpy(np.float32),
        "car1": signed_log(df["car_minus1_plus1"].clip(-clip, clip)).to_numpy(np.float32),
        "car5": signed_log(df["car_minus1_plus5"].clip(-clip, clip)).to_numpy(np.float32),
        "volume_change": signed_log(volume.fillna(0)).to_numpy(np.float32),
        "volume_present": volume.notna().to_numpy(),           # blank when baseline volume was zero
        "reaction_onehot": np.stack([(df["reaction_category"] == c).to_numpy(np.float32)
                                     for c in REACTION_CLASSES], axis=1),
    }

    # ---------------------------------------------------------------- 6. targets (raw, never clipped)
    target_car1 = df["car_minus1_plus1"].to_numpy(np.float32)
    target_class = df["reaction_category"].map(REACTION_CLASSES.index).to_numpy(np.int64)

    data = {
        "event_x": event_x.to_numpy(np.float32), "event_columns": list(event_x.columns),
        "company_x": company_x.to_numpy(np.float32), "company_columns": list(company_x.columns),
        **outcome,
        "reaction_classes": REACTION_CLASSES,
        "target_car1": target_car1, "target_class": target_class,
        "trading_day": trading_day, "company_start": company_start,
        "event_date": df["report_date"].to_numpy(), "ticker": df["ticker"].to_numpy(),
        "event_id": df["event_id"].to_numpy(),
        "dropped_unreliable": unreliable,
    }
    with open(cfg.EVENTS_PKL, "wb") as f:
        pickle.dump(data, f)

    print(f"dropped {unreliable:,} events with price_data_unreliable == True")
    print(f"{len(df):,} events from {df['ticker'].nunique():,} companies, "
          f"{len(calendar):,} trading days ({str(calendar[0])[:10]} to {str(calendar[-1])[:10]})")
    print(f"event_x: {event_x.shape[1]} columns | company_x: {company_x.shape[1]} columns"
          f"{' (look-ahead features dropped)' if cfg.DROP_LOOKAHEAD_FEATURES else ''}")
    print(f"saved {cfg.EVENTS_PKL} ({cfg.EVENTS_PKL.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
