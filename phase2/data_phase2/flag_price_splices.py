"""
EECS 4412 Phase 2 -- find spliced prices in the Yahoo cache and flag the events they touch.

The bug (first seen on NFE): around a reverse split Yahoo's history mixes split-adjusted and
unadjusted closes, so the price jumps by the split ratio and jumps back a few sessions later.

  1. Round trip: for every cached company, a day with |log return| > MIN_LOG_MOVE and an opposite
     move within MAX_SESSIONS_APART sessions that cancels it to within MAX_OFFSET_ERROR. Both days
     are flagged. A one-way jump is not flagged, even when it looks like the same bug.
  2. Non-positive close: a return using a close <= 0 has no log and an inverted sign, so those
     days are flagged too (FLAG_NONPOSITIVE). The ratio close/previous-1 reads a fall from -1.36
     to -1.45 as +6.6%.
  3. Events: an event is flagged when a flagged date falls on its trading days -1..+5. CARs sum
     daily abnormal returns over those days, so this covers CAR[-1,+1] and CAR[-1,+5].
  4. One-way suspects, listed but NOT flagged: an extreme move with no reversal that matches a split
     on record or came on collapsing volume. On a split date the bug can be a single seam, which a
     round-trip test cannot see (NFE moved x42 on its own 2026-09-14 split date).

Price files and every existing value stay as they are. market_reactions.csv gains one column,
price_data_unreliable; rerun build_full_dataset.py afterwards to carry it into full_dataset.
pipeline_phase2.py build rewrites market_reactions.csv without it, so rerun this after a build.

Run from inside data_phase2/ with the interpreter that has yfinance (same as the pipeline):
    python flag_price_splices.py
Writes raw/price_splice_flags.csv, raw/price_unreliable_events.csv, raw/price_one_way_suspects.csv,
raw/price_splice_split_records.json, processed/price_splice_log.txt and the new column in
processed/market_reactions.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROC, RAW, PRICES = HERE / "processed", HERE / "raw", HERE / "cache" / "prices"
PHASE1_COMPANIES = HERE.parents[1] / "phase1" / "data" / "processed" / "companies.csv"
sys.path.append(str(HERE.parent / "network"))
from tier0_config import TRAIN_END, VAL_END  # noqa: E402

MIN_LOG_MOVE = 1.5        # at least a x4.48 move; below this most hits are penny-price bounce or real spikes
MAX_SESSIONS_APART = 5
MAX_OFFSET_ERROR = 0.10   # |move + reversal| / |move|
FLAG_NONPOSITIVE = True
WINDOW = (-1, 5)          # trading days a CAR[-1,+5] sums over; CAR[-1,+1] is the first three
SHORT_WINDOW = (-1, 1)
ROUND_RATIOS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 150, 200, 250, 500, 1000]
SPLIT_FACTOR_TOLERANCE = 1.25   # a split "matches" when its factor is within 25% of the move
SPLIT_DAYS_APART = 90           # and it falls within this many calendar days of the move
COLLAPSED_VOLUME = 0.5          # one-way suspect: volume below this share of the prior 20-day median
READ = dict(keep_default_na=False, na_values=[""])   # real tickers such as "NA" stay strings
LOG_LINES = []


def say(text=""):
    print(text, flush=True)
    LOG_LINES.append(str(text))


def round_trip_positions(log_returns):
    """Row positions of both legs of every round trip, with the leg each extreme day paired with."""
    legs = {}
    for i in np.flatnonzero(np.abs(log_returns) > MIN_LOG_MOVE):
        nearby = [j for j in range(max(i - MAX_SESSIONS_APART, 0), min(i + MAX_SESSIONS_APART + 1, len(log_returns)))
                  if log_returns[i] * log_returns[j] < 0]
        if not nearby:
            continue
        errors = [abs(log_returns[i] + log_returns[j]) / abs(log_returns[i]) for j in nearby]
        if min(errors) <= MAX_OFFSET_ERROR:
            legs[i] = min(errors)
            legs.setdefault(nearby[int(np.argmin(errors))], min(errors))
    return legs


def episodes(positions):
    """Split sorted row positions into runs no more than MAX_SESSIONS_APART sessions apart."""
    breaks = np.flatnonzero(np.diff(positions) > MAX_SESSIONS_APART) + 1
    return np.split(np.asarray(positions), breaks)


def volume_ratio(volume, days, start):
    """Median volume on the given days over the median of the 20 sessions before start."""
    baseline = np.nanmedian(volume[max(start - 20, 0):start]) if start else np.nan
    return round(float(np.nanmedian(volume[days]) / baseline), 3) if baseline and baseline > 0 else np.nan


def scan_company(ticker, calendar):
    """One row per episode, one per one-way move, and the flagged calendar positions by kind."""
    prices = pd.read_csv(PRICES / f"t_{ticker}.csv.gz", index_col=0, parse_dates=True)
    if prices.empty:
        return [], [], [], [], False
    # same calendar as pipeline_phase2.reactions_for_ticker, so a date here is a date in the CARs
    prices = prices.reindex(calendar).dropna(subset=["Close"])
    close, volume = prices["Close"].to_numpy(), prices["Volume"].to_numpy()
    if len(close) < 2:
        return [], [], [], [], False
    dates = prices.index
    positive = np.where(close > 0, close, np.nan)
    log_returns = np.r_[np.nan, np.log(positive[1:] / positive[:-1])]
    legs = round_trip_positions(log_returns)

    rows = []
    if legs:
        for run in episodes(sorted(legs)):
            start, end = run[0], run[-1]
            multiplier = float(np.exp(np.median(np.abs(log_returns[run]))))
            rows.append({
                "ticker": ticker, "kind": "round_trip", "first_date": dates[start].date(), "last_date": dates[end].date(),
                "flagged_dates": ";".join(str(dates[i].date()) for i in run), "days": len(run),
                "multiplier": round(multiplier, 2), "nearest_round_ratio": min(ROUND_RATIOS, key=lambda r: abs(np.log(r / multiplier))),
                "worst_offset_error": round(max(legs[i] for i in run), 3),
                "close_before": close[start - 1], "close_on_first_jump": close[start], "close_after": close[end],
                "volume_vs_prior_20d": volume_ratio(volume, run, start),
            })

    bad = close <= 0
    nonpositive = np.flatnonzero(bad | np.r_[False, bad[:-1]])    # a return is wrong if either close is
    if len(nonpositive):
        for run in np.split(nonpositive, np.flatnonzero(np.diff(nonpositive) > 1) + 1):
            rows.append({"ticker": ticker, "kind": "nonpositive_close", "first_date": dates[run[0]].date(),
                         "last_date": dates[run[-1]].date(), "flagged_dates": f"every session in this range ({len(run)})",
                         "days": len(run), "close_before": close[run[0] - 1] if run[0] else np.nan,
                         "close_on_first_jump": close[run[0]], "close_after": close[run[-1]]})

    to_calendar = calendar.get_indexer(dates)
    one_way = [{"ticker": ticker, "date": dates[i].date(), "first_date": dates[i].date(), "position": to_calendar[i],
                "multiplier": round(float(np.exp(abs(log_returns[i]))), 2), "close_before": close[i - 1], "close_after": close[i],
                "volume_vs_prior_20d": volume_ratio(volume, [i], i)}
               for i in np.flatnonzero(np.abs(log_returns) > MIN_LOG_MOVE) if i not in legs]
    return rows, one_way, to_calendar[sorted(legs)].tolist(), to_calendar[nonpositive].tolist(), True


def split_records(tickers):
    """Yahoo's split history for the flagged tickers, saved so a rerun needs no network."""
    path = RAW / "price_splice_split_records.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    missing = [t for t in tickers if t not in records]
    if missing:
        import yfinance as yf     # only the fetch needs it
        for ticker in missing:
            records[ticker] = [[str(d.date()), float(f)] for d, f in yf.Ticker(ticker).splits.items()]
        path.write_text(json.dumps(records, indent=0, sort_keys=True))
    return records


def matching_split(row, records):
    """The first split on record close to this episode in both size and date, or blank."""
    if pd.isna(row["multiplier"]):
        return ""
    for day, factor in records.get(row["ticker"], []):
        size = max(factor, 1 / factor)
        near = abs((pd.Timestamp(day) - pd.Timestamp(row["first_date"])).days) <= SPLIT_DAYS_APART
        if near and size > 1 and abs(np.log(size / row["multiplier"])) < np.log(SPLIT_FACTOR_TOLERANCE):
            return f"1-for-{size:g} on {day}" if factor < 1 else f"{size:g}-for-1 on {day}"
    return ""


def flag_events(events, flagged, calendar):
    """Boolean arrays: does each event's window hold a flagged date, per window in (WINDOW, SHORT_WINDOW)."""
    position = calendar.get_indexer(events["trading_date"])
    assert (position >= 0).all(), "every trading_date must be on the ^GSPC calendar"
    wide, short = np.zeros(len(events), bool), np.zeros(len(events), bool)
    for ticker, rows in events.groupby("ticker").indices.items():
        days = np.sort(np.asarray(flagged.get(ticker, []), dtype=int))
        if len(days):
            for hits, (low, high) in ((wide, WINDOW), (short, SHORT_WINDOW)):
                counts = (np.searchsorted(days, position[rows] + high, side="right")
                          - np.searchsorted(days, position[rows] + low, side="left"))
                hits[rows] = counts > 0
    return wide, short


def add_flag_column(flag_by_event):
    """Add the column by rewriting the file as text, so every existing value stays byte for byte."""
    path = PROC / "market_reactions.csv"
    with open(path, newline="") as f:
        newline = "\r\n" if f.readline().endswith("\r\n") else "\n"
    table = pd.read_csv(path, dtype=str, keep_default_na=False).drop(columns="price_data_unreliable", errors="ignore")
    table["price_data_unreliable"] = table["event_id"].map(flag_by_event).map({True: "True", False: "False"})
    assert table["price_data_unreliable"].notna().all()
    table.to_csv(path, index=False, lineterminator=newline)


def main():
    calendar = pd.read_csv(PRICES / "t_IDX_GSPC.csv.gz", index_col=0, parse_dates=True).index
    companies = set(pd.read_csv(PROC / "companies.csv", **READ)["ticker"])
    phase1 = set(pd.read_csv(PHASE1_COMPANIES, **READ)["ticker"])
    cached = sorted(p.name[2:-7] for p in PRICES.glob("t_*.csv.gz") if p.name != "t_IDX_GSPC.csv.gz")

    rows, one_way, round_trip_days, nonpositive_days, scanned = [], [], {}, {}, 0
    for ticker in cached:
        found, moves, legs, nonpositive, usable = scan_company(ticker, calendar)
        rows += found
        one_way += [m for m in moves if ticker in companies]
        scanned += usable
        if legs:
            round_trip_days[ticker] = legs
        if nonpositive:
            nonpositive_days[ticker] = nonpositive

    flags = pd.DataFrame(rows).sort_values(["ticker", "first_date"]).reset_index(drop=True)
    flags.insert(1, "in_companies_table", flags["ticker"].isin(companies))
    flags.insert(2, "in_phase1_sample", flags["ticker"].isin(phase1))
    one_way = pd.DataFrame(one_way)
    records = split_records(sorted(set(flags.loc[flags["kind"] == "round_trip", "ticker"]) | set(one_way["ticker"])))
    flags["split_on_record_matching_move"] = flags.apply(matching_split, axis=1, records=records)
    flags["any_split_on_record"] = flags["ticker"].map(lambda t: "yes" if records.get(t) else "no").where(flags["kind"] == "round_trip", "")
    flags.to_csv(RAW / "price_splice_flags.csv", index=False)

    events = pd.read_csv(PROC / "events.csv", usecols=["event_id", "ticker", "report_date", "event_category"], **READ)
    reactions = pd.read_csv(PROC / "market_reactions.csv", usecols=["event_id", "trading_date", "daily_return",
                            "abnormal_return", "car_minus1_plus1", "car_minus1_plus5"], parse_dates=["trading_date"])
    events = events.merge(reactions, on="event_id", validate="one_to_one")
    round_wide, round_short = flag_events(events, round_trip_days, calendar)
    negative_wide, negative_short = flag_events(events, nonpositive_days, calendar)
    unreliable = round_wide | negative_wide if FLAG_NONPOSITIVE else round_wide
    add_flag_column(dict(zip(events["event_id"], unreliable)))

    report = pd.to_datetime(events["report_date"])
    events["tier0_split"] = np.select([report < TRAIN_END, report < VAL_END], ["train", "validation"], "test")
    events["in_phase1_sample"] = events["ticker"].isin(phase1)
    events["round_trip_in_car1"], events["round_trip_in_car5"] = round_short, round_wide
    events["nonpositive_in_car1"], events["nonpositive_in_car5"] = negative_short, negative_wide
    flagged_events = events[unreliable].sort_values(["ticker", "trading_date"])
    flagged_events.to_csv(RAW / "price_unreliable_events.csv", index=False)

    one_way["split_on_record_matching_move"] = one_way.apply(matching_split, axis=1, records=records)
    suspects = one_way[(one_way["split_on_record_matching_move"] != "")
                       | (one_way["volume_vs_prior_20d"] < COLLAPSED_VOLUME)].drop(columns="first_date").copy()
    events["position"] = calendar.get_indexer(events["trading_date"])
    by_ticker = dict(tuple(events.groupby("ticker")))
    # an event's window holds the move when the event's day 0 is 5 sessions before it to 1 session after it
    suspects["event_ids_in_window"] = [
        ";".join(by_ticker[s.ticker].loc[by_ticker[s.ticker]["position"].between(s.position - WINDOW[1], s.position - WINDOW[0]), "event_id"])
        if s.ticker in by_ticker else "" for s in suspects.itertuples()]
    suspects.drop(columns="position").sort_values(["ticker", "date"]).to_csv(RAW / "price_one_way_suspects.csv", index=False)
    suspect_ids = {e for ids in suspects["event_ids_in_window"] if ids for e in ids.split(";")}

    in_table = flags[flags["in_companies_table"]]
    trips = flags[flags["kind"] == "round_trip"]
    say(f"Settings: |log return| > {MIN_LOG_MOVE} (x{np.exp(MIN_LOG_MOVE):.2f}), reversal within "
        f"{MAX_SESSIONS_APART} sessions, offset error <= {MAX_OFFSET_ERROR:.0%}, non-positive closes flagged: {FLAG_NONPOSITIVE}")
    say(f"Price files scanned: {scanned:,} of {len(cached):,} cached tickers ({len(companies):,} in companies.csv)")
    say(f"Round-trip companies: {trips['ticker'].nunique()} in the cache, "
        f"{trips.loc[trips['in_companies_table'], 'ticker'].nunique()} in companies.csv")
    say(f"Non-positive-close companies: {flags.loc[flags['kind'] == 'nonpositive_close', 'ticker'].nunique()} in the cache, "
        f"{in_table.loc[in_table['kind'] == 'nonpositive_close', 'ticker'].nunique()} in companies.csv")
    say(f"Flagged companies, either kind: {flags['ticker'].nunique()} in the cache, {in_table['ticker'].nunique()} in companies.csv, "
        f"{flagged_events['ticker'].nunique()} with a flagged event")
    say(f"Events: {len(events):,}; price_data_unreliable = True: {unreliable.sum():,} "
        f"(round trip {round_wide.sum():,}, non-positive close {negative_wide.sum():,}, both {(round_wide & negative_wide).sum():,})")
    say(f"Flag inside CAR[-1,+1] itself: {(round_short | negative_short)[unreliable].sum():,}; only in days +2..+5: "
        f"{(~(round_short | negative_short))[unreliable].sum():,}")
    say(f"Tier 0 split of flagged events: {flagged_events['tier0_split'].value_counts().to_dict()}")
    say(f"Flagged events in the Phase 1 sample: {flagged_events['in_phase1_sample'].sum()}")
    say(f"One-way suspects, not flagged: {len(suspects)} moves at {suspects['ticker'].nunique()} companies "
        f"({(suspects['split_on_record_matching_move'] != '').sum()} match a split), touching {len(suspect_ids)} events; "
        f"in the Phase 1 sample: {suspects['ticker'].isin(phase1).sum()}")
    (PROC / "price_splice_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
