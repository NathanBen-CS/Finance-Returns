"""
EECS 4412 Phase 2 -- corporate events & abnormal returns at scale.

Same event-study methodology as Phase 1 (data/pipeline.py): SEC 8-K filings
joined to market-adjusted abnormal returns vs ^GSPC, CAR[-1,+1], CAR[-1,+5],
reaction_category buckets, event_category mapping and filing_timeliness
buckets. Those functions are IMPORTED from data/pipeline.py, not copied, so the
two phases cannot drift. What changes is the data-pull step:

  * window 2016-01-01 .. today (cut later if per-year data quality degrades)
  * universe = every NYSE/Nasdaq common stock in SEC's own ticker list, no index
    or market-cap filter, each verified against its own SEC submissions record
  * stratification dimensions measured from real data: sector, market-cap tier,
    and trailing 1-year realized volatility tier

Stages (each resumable -- per-company checkpoints under ./cache):
  universe   SEC ticker list -> raw/universe_candidates.csv
  sec        data.sec.gov/submissions per CIK -> cache/sec/<cik>.json.gz
  prices     Yahoo daily history, batched     -> cache/prices/t_<ticker>.csv.gz
  info       Yahoo fundamentals per ticker    -> cache/info/t_<ticker>.json
  build      tiers, events, reactions, logs   -> processed/*.csv, run_log.txt

Usage:  python pipeline_phase2.py <universe|sec|prices|info|build|all> [--limit N]
        --limit N runs on N evenly spaced candidates (smoke test).
Requires: pandas, numpy, requests, yfinance (run with the interpreter that has
yfinance installed; the project .venv does not).
"""
import argparse
import gzip
import json
import math
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import yfinance.shared as yf_shared  # per-ticker download errors (rate limits vs no data)

ROOT = Path(__file__).resolve().parent
RAW, PROC, CACHE = ROOT / "raw", ROOT / "processed", ROOT / "cache"
REFERENCE = ROOT.parent / "reference"
PHASE1 = ROOT.parent / "data"

# Phase 1 methodology, imported read-only. No bytecode is written so the frozen
# Phase 1 folder is never modified.
sys.dont_write_bytecode = True
sys.path.insert(0, str(PHASE1))
import pipeline as P1  # noqa: E402

WINDOW_START = date(2016, 1, 1)
PRICE_START = date(2015, 10, 1)        # >= 45 calendar days of history before the first event
SEC_RATE = 5.0                         # requests/second; SEC's stated ceiling is 10
SEC_THREADS = 4
YF_BATCH = 100
VOL_DAYS, VOL_MIN_OBS = 252, 240       # trailing window and minimum valid returns in it
STALE_DAYS = 5                         # last close must be within 5 trading days of ^GSPC's last
MAX_ZERO_VOLUME = 0.50                 # > half the trailing year with no trades = stale quotes
EXTREME_MOVE = 1.0                     # |daily_return| or |CAR| above 100% is listed for review
YEAR_MIN_TAGGED, YEAR_MIN_PRICED = 0.95, 0.80
NONCOMMON = re.compile(r"-(P[A-Z]?|WT|WS|W|UN|U|RT|RI|R)$")  # preferred, warrants, units, rights
SIZE_LABELS = ["Small", "Mid", "Large", "Mega-cap"]   # Phase 1's labels
VOL_LABELS = ["Low", "Medium", "High", "Very High"]
SELECTION = "stratified_universe_census"


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def log(msg, logfile=None):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(CACHE / "logs" / logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def safe_name(ticker):
    # prefix avoids Windows reserved names (CON, PRN, AUX, NUL ...) used as tickers
    return "t_" + ticker.replace("^", "IDX_")


def write_gz_json(path, obj):
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    tmp.replace(path)


def read_gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class Stats:
    """Cumulative per-stage counters persisted across resumed runs."""

    def __init__(self, stage):
        self.path = CACHE / f"stats_{stage}.json"
        self.lock = threading.Lock()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.t0 = time.time()

    def inc(self, key, n=1):
        with self.lock:
            self.data[key] = self.data.get(key, 0) + n

    def save(self):
        with self.lock:
            self.data["elapsed_s"] = self.data.get("elapsed_s_prev", 0) + (time.time() - self.t0)
            self.path.write_text(json.dumps(self.data, indent=1))

    def close(self):
        self.save()
        with self.lock:
            self.data["elapsed_s_prev"] = self.data["elapsed_s"]
            self.path.write_text(json.dumps(self.data, indent=1))


class RateLimiter:
    def __init__(self, per_sec):
        self.interval, self.lock, self.next = 1.0 / per_sec, threading.Lock(), time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.interval
        time.sleep(max(0.0, slot - now))

    def pause(self, seconds):  # global back-off after a 429/403
        with self.lock:
            self.next = max(self.next, time.monotonic() + seconds)


def candidates(limit=None):
    # keep_default_na=False: real tickers such as "NA" must not be parsed as missing values
    df = pd.read_csv(RAW / "universe_candidates.csv", dtype=str, keep_default_na=False)
    df["cik"] = df["cik"].astype(int)
    if limit:
        idx = np.linspace(0, len(df) - 1, limit).round().astype(int)
        df = df.iloc[sorted(set(idx))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 1: universe
# ---------------------------------------------------------------------------
def stage_universe():
    ex_path = RAW / "sec_company_tickers_exchange.json"
    if not ex_path.exists():
        r = requests.get("https://www.sec.gov/files/company_tickers_exchange.json",
                         headers=P1.SEC_HEADERS, timeout=60)
        r.raise_for_status()
        ex_path.write_bytes(r.content)
    ex = json.loads(ex_path.read_text())
    exchange = {(int(c), t): e for c, _, t, e in ex["data"]}

    base = json.loads((REFERENCE / "company_tickers.json").read_text())
    rows = [dict(rank=int(k), cik=v["cik_str"], ticker=v["ticker"], title=v["title"])
            for k, v in sorted(base.items(), key=lambda kv: int(kv[0]))]
    df = pd.DataFrame(rows)
    df["exchange_sec"] = [exchange.get((c, t), "<missing>") for c, t in zip(df["cik"], df["ticker"])]

    excluded, keep = [], []
    tickers_by_cik = df.groupby("cik")["ticker"].apply(set).to_dict()
    for r in df.itertuples():
        if r.exchange_sec == "<missing>":
            reason = "not in SEC exchange file (ticker changed/delisted since list date)"
        elif pd.isna(r.exchange_sec):
            reason = "SEC exchange file lists no exchange"
        elif r.exchange_sec not in ("NYSE", "Nasdaq"):
            reason = f"exchange {r.exchange_sec}"
        elif NONCOMMON.search(r.ticker):
            reason = "non-common security suffix (preferred/warrant/unit/right)"
        elif (len(r.ticker) == 5 and r.ticker[-1] in "WUR"
              and r.ticker[:-1] in tickers_by_cik[r.cik]):
            reason = "Nasdaq warrant/unit/right of a listed common"
        else:
            reason = None
        (excluded if reason else keep).append({**r._asdict(), "reason": reason})
    kept = pd.DataFrame(keep)
    dup = kept.duplicated("cik", keep="first")  # one primary common per company
    for r in kept[dup].itertuples():
        excluded.append({**r._asdict(), "reason": "secondary ticker of same CIK (first-listed kept)"})
    kept = kept[~dup].drop(columns=["Index", "reason"], errors="ignore")

    kept.to_csv(RAW / "universe_candidates.csv", index=False)
    exc = pd.DataFrame(excluded).drop(columns=["Index"], errors="ignore")
    exc.to_csv(RAW / "universe_excluded.csv", index=False)
    summary = {"reference_rows": len(df), "candidates": len(kept),
               "excluded": exc["reason"].value_counts().to_dict()}
    (CACHE / "stats_universe.json").write_text(json.dumps(summary, indent=1))
    log(f"universe: {len(df)} SEC tickers -> {len(kept)} candidate companies "
        f"({len(exc)} rows excluded: {summary['excluded']})", "universe.log")


# ---------------------------------------------------------------------------
# Stage 2: SEC submissions, one company at a time
# ---------------------------------------------------------------------------
SEC_LIMIT = RateLimiter(SEC_RATE)
_local = threading.local()


def sec_get(url, stats):
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update(P1.SEC_HEADERS)
    for attempt in range(1, 8):
        SEC_LIMIT.wait()
        try:
            r = _local.session.get(url, timeout=30)
        except requests.RequestException:
            stats.inc("network_errors")
            time.sleep(2 ** attempt)
            continue
        stats.inc("requests")
        stats.inc(f"http_{r.status_code}")
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429, 500, 502, 503, 504):
            wait = min(120, (10 if r.status_code in (403, 429) else 2) * 2 ** (attempt - 1))
            stats.inc("backoffs")
            SEC_LIMIT.pause(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    raise RuntimeError(f"gave up after retries: {url}")


def eightk_rows(block, source_cik):
    df = P1.filings_frame(block)
    if "acceptanceDateTime" in block:
        df["acceptanceDateTime"] = block["acceptanceDateTime"]
    df = df[df["form"].isin(["8-K", "8-K/A"]) & (df["filingDate"] >= "2015-12-01")].copy()
    df["source_cik"] = source_cik
    return df


def fetch_company(cik, stats):
    """Full 8-K list since 2016 for one CIK from its own submissions record."""
    sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", stats)
    if sub is None:
        return None, []
    rec = {k: sub.get(k) for k in ("name", "entityType", "sic", "sicDescription", "exchanges", "tickers")}
    frames = [eightk_rows(sub["filings"]["recent"], cik)]
    pages = 0
    eligible = rec["entityType"] == "operating" and str(rec["sic"]) != "6770"
    recent_dates = sub["filings"]["recent"].get("filingDate", [])
    if eligible and recent_dates and min(recent_dates) > WINDOW_START.isoformat():
        for f in sub["filings"].get("files", []):
            if f["filingTo"] >= WINDOW_START.isoformat():
                page = sec_get(f"https://data.sec.gov/submissions/{f['name']}", stats)
                pages += 1
                if page:
                    frames.append(eightk_rows(page, cik))
    rec["pages_fetched"] = pages
    return rec, frames


def sec_company(c, stats):
    path = CACHE / "sec" / f"{c.cik}.json.gz"
    if path.exists():
        return "cached"
    rec, frames = fetch_company(int(c.cik), stats)
    if rec is None:
        write_gz_json(path, {"cik": int(c.cik), "ticker": c.ticker, "status": "not_found"})
        return "not_found"
    for extra in P1.EXTRA_CIKS.get(c.ticker, []):  # predecessor registrant (Phase 1: XOM)
        _, more = fetch_company(extra, stats)
        frames += more
    filings = pd.concat(frames, ignore_index=True).drop_duplicates("accessionNumber")
    in_window = filings[(filings["form"] == "8-K") &
                        (filings["reportDate"].where(filings["reportDate"] != "", filings["filingDate"])
                         >= WINDOW_START.isoformat())]
    if rec["entityType"] != "operating":
        status = "not_operating"
    elif str(rec["sic"]) == "6770":
        status = "blank_check"
    elif in_window.empty:
        status = "no_8k_in_window"
    else:
        status = "ok"
    write_gz_json(path, {"cik": int(c.cik), "ticker": c.ticker, "status": status, **rec,
                         "filings": filings.to_dict("list")})
    return status


def stage_sec(limit=None):
    (CACHE / "sec").mkdir(parents=True, exist_ok=True)
    cands = candidates(limit)
    todo = [c for c in cands.itertuples() if not (CACHE / "sec" / f"{c.cik}.json.gz").exists()]
    stats = Stats("sec")
    log(f"sec: {len(cands)} candidates, {len(cands) - len(todo)} already checkpointed, "
        f"{len(todo)} to fetch at <= {SEC_RATE:.0f} req/s", "sec.log")
    done, t0, outcomes = 0, time.time(), Counter()
    with ThreadPoolExecutor(SEC_THREADS) as pool:
        futs = {pool.submit(sec_company, c, stats): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                outcomes[fut.result()] += 1
            except Exception as e:  # leave unwritten so a re-run retries it
                outcomes["error"] += 1
                log(f"sec: {c.ticker} (CIK {c.cik}) failed: {e}", "sec.log")
            done += 1
            if done % 100 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                log(f"sec: {done}/{len(todo)} companies, {stats.data.get('requests', 0)} requests, "
                    f"{rate:.2f} co/s, ETA {(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min, "
                    f"{dict(outcomes)}", "sec.log")
                stats.save()
    stats.close()


# ---------------------------------------------------------------------------
# Stage 3: Yahoo prices, batched
# ---------------------------------------------------------------------------
def price_path(ticker):
    return CACHE / "prices" / f"{safe_name(ticker)}.csv.gz"


def nohist_path(ticker):
    return CACHE / "prices" / f"{safe_name(ticker)}.nohistory.json"


def sec_status(cik):
    p = CACHE / "sec" / f"{cik}.json.gz"
    return read_gz_json(p)["status"] if p.exists() else None


def save_prices(ticker, df):
    df = df[["Close", "Volume"]].dropna(subset=["Close"])
    if df.empty:
        return False
    df.to_csv(price_path(ticker), compression="gzip")
    return True


def stage_prices(limit=None):
    (CACHE / "prices").mkdir(parents=True, exist_ok=True)
    stats = Stats("prices")
    end = (date.today() + timedelta(days=1)).isoformat()
    if not price_path("^GSPC").exists():
        idx = yf.download("^GSPC", start=PRICE_START.isoformat(), end=end, auto_adjust=True, progress=False)
        stats.inc("requests")
        if isinstance(idx.columns, pd.MultiIndex):
            idx.columns = idx.columns.get_level_values(0)
        save_prices("^GSPC", idx)

    def pending(t):
        if price_path(t).exists():
            return False
        if nohist_path(t).exists():
            return json.loads(nohist_path(t).read_text())["attempts"] < 2
        return True

    cands = candidates(limit)
    # skip companies SEC has already ruled out; unknown status (not fetched yet) is kept
    cands = cands[[sec_status(c) in (None, "ok") for c in cands["cik"]]]
    for pass_no in range(1, 4):  # whole-batch Yahoo failures are retried in later passes
        todo = [t for t in cands["ticker"] if pending(t)]
        if not todo:
            break
        log(f"prices: pass {pass_no}: {len(todo)} tickers to download in batches of {YF_BATCH}", "prices.log")
        for b in range(0, len(todo), YF_BATCH):
            batch = todo[b: b + YF_BATCH]
            for attempt in range(1, 6):
                try:
                    data = yf.download(batch, start=PRICE_START.isoformat(), end=end, auto_adjust=True,
                                       group_by="ticker", threads=True, progress=False)
                    stats.inc("requests")
                    break
                except Exception as e:
                    stats.inc("backoffs")
                    log(f"prices: batch {b // YF_BATCH} attempt {attempt} failed ({e}); backing off", "prices.log")
                    time.sleep(30 * attempt)
            else:
                continue
            errors = dict(getattr(yf_shared, "_ERRORS", {}) or {})
            rate_limited = any("rate" in str(m).lower() or "too many" in str(m).lower() for m in errors.values())
            # A batch where nothing came back is a Yahoo-side failure (observed: every ticker
            # "possibly delisted; no timezone found", live stocks included) -> retry, never mark.
            if len(batch) > 1 and not any(
                    isinstance(data.columns, pd.MultiIndex) and t in data.columns.get_level_values(0)
                    and data[t]["Close"].notna().any() for t in batch):
                stats.inc("empty_batches")
                log(f"prices: batch {b // YF_BATCH + 1} returned nothing for all {len(batch)} tickers "
                    f"-> treated as transient, left for retry", "prices.log")
                time.sleep(30)
                continue
            got = 0
            for t in batch:
                sub = None
                if isinstance(data.columns, pd.MultiIndex) and t in data.columns.get_level_values(0):
                    sub = data[t]
                elif not isinstance(data.columns, pd.MultiIndex) and len(batch) == 1:
                    sub = data
                if sub is not None and save_prices(t, sub):
                    got += 1
                    nohist_path(t).unlink(missing_ok=True)  # an earlier miss was transient
                    continue
                if rate_limited and t in errors:
                    continue  # transient; retried on the next run
                prev = json.loads(nohist_path(t).read_text())["attempts"] if nohist_path(t).exists() else 0
                nohist_path(t).write_text(json.dumps({"attempts": prev + 1, "error": str(errors.get(t, "empty"))}))
            stats.inc("tickers_with_history", got)
            log(f"prices: batch {b // YF_BATCH + 1}/{math.ceil(len(todo) / YF_BATCH)} -> "
                f"{got}/{len(batch)} with history{' (rate-limited)' if rate_limited else ''}", "prices.log")
            stats.save()
            if rate_limited:
                time.sleep(60)
            time.sleep(1.0)

    # Confirm every suspected no-history ticker with its own single-ticker download
    # before ruling it out; batch misses are retried rather than trusted.
    suspects = [t for t in cands["ticker"]
                if not price_path(t).exists() and nohist_path(t).exists()
                and not json.loads(nohist_path(t).read_text()).get("confirmed")]
    log(f"prices: confirming {len(suspects)} suspected no-history tickers one at a time", "prices.log")
    for i, t in enumerate(suspects, 1):
        try:
            d = yf.download(t, start=PRICE_START.isoformat(), end=end, auto_adjust=True, progress=False)
            stats.inc("requests")
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if not d.empty and save_prices(t, d):
                nohist_path(t).unlink()
                stats.inc("recovered_single")
            else:
                err = str((getattr(yf_shared, "_ERRORS", {}) or {}).get(t, "empty"))
                nohist_path(t).write_text(json.dumps({"attempts": 99, "confirmed": True, "error": err}))
        except Exception as e:
            log(f"prices: single download {t} failed ({e}); left for retry", "prices.log")
        if i % 50 == 0:
            log(f"prices: confirmed {i}/{len(suspects)}, recovered so far "
                f"{stats.data.get('recovered_single', 0)}", "prices.log")
            stats.save()
        time.sleep(0.3)
    stats.close()


def load_prices(ticker):
    p = price_path(ticker)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)


def trailing_vol(px, idx):
    """Annualized std of daily returns over the last VOL_DAYS ^GSPC sessions. None if the
    series is stale, mostly non-trading, or has fewer than VOL_MIN_OBS valid returns."""
    cal = idx.index[-(VOL_DAYS + 1):]
    if px.index.max() < idx.index[-1 - STALE_DAYS]:
        return None, "stale price series (no recent closes)"
    close = px["Close"].reindex(cal)
    zero_vol = float((px["Volume"].reindex(cal) == 0).mean())
    if zero_vol > MAX_ZERO_VOLUME:
        # quotes without trades: repeated closes fake a near-zero volatility
        return None, f"non-trading: zero volume on > {MAX_ZERO_VOLUME:.0%} of trailing-year sessions"
    ret = (close / close.shift(1) - 1).iloc[1:]
    n = int(ret.notna().sum())
    if n < VOL_MIN_OBS:
        return None, f"insufficient trailing-year history (< {VOL_MIN_OBS} daily returns)"
    return float(ret.std(ddof=1) * math.sqrt(252)), None


# ---------------------------------------------------------------------------
# Stage 4: Yahoo fundamentals, one ticker at a time
# ---------------------------------------------------------------------------
INFO_FIELDS = ("quoteType", "exchange", "marketCap", "totalRevenue", "fullTimeEmployees",
               "sector", "industry", "longName")


def info_path(ticker):
    return CACHE / "info" / f"{safe_name(ticker)}.json"


def fetch_info(ticker, stats):
    for attempt in range(1, 6):
        try:
            tk = yf.Ticker(ticker)
            info = tk.get_info()
            stats.inc("requests")
            if not info or ("marketCap" not in info and "exchange" not in info and "quoteType" not in info):
                raise ValueError("empty info payload")
            rec = {k: info.get(k) for k in INFO_FIELDS}
            rec["market_cap_source"] = "info"
            if rec["marketCap"] is None:  # Phase 1 fallback: shares x price, else fast_info
                rec["marketCap"] = P1.derive_market_cap(ticker, tk, info)
                rec["market_cap_source"] = "derived" if rec["marketCap"] else None
            return rec
        except Exception as e:
            msg = str(e)
            limited = "rate" in msg.lower() or "too many" in msg.lower() or "429" in msg
            stats.inc("rate_limited" if limited else "errors")
            if not limited and attempt >= 3:
                return {"error": msg[:200]}
            time.sleep(min(300, (60 if limited else 3) * attempt))
    return {"error": "gave up after retries"}


def stage_info(limit=None):
    (CACHE / "info").mkdir(parents=True, exist_ok=True)
    stats = Stats("info")
    idx = load_prices("^GSPC")
    todo = []
    for c in candidates(limit).itertuples():
        if info_path(c.ticker).exists() or sec_status(c.cik) != "ok":
            continue
        px = load_prices(c.ticker)
        if px is None or trailing_vol(px, idx)[0] is None:
            continue
        todo.append(c.ticker)
    log(f"info: {len(todo)} tickers to fetch", "info.log")
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        info_path(t).write_text(json.dumps(fetch_info(t, stats)))
        time.sleep(0.4)
        if i % 100 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            log(f"info: {i}/{len(todo)}, {rate:.2f}/s, ETA {(len(todo) - i) / rate / 60:.0f} min, "
                f"rate-limited={stats.data.get('rate_limited', 0)} errors={stats.data.get('errors', 0)}",
                "info.log")
            stats.save()
    stats.close()


# ---------------------------------------------------------------------------
# Stage 5: build -- reactions use Phase 1's exact formulas
# ---------------------------------------------------------------------------
def reactions_for_ticker(px, idx, events):
    """Phase 1 build_reactions() per-event logic, on the ^GSPC trading calendar so a day
    the stock did not trade is a gap (-> dropped), never silently bridged."""
    j = idx[["Close"]].rename(columns={"Close": "MktClose"}).join(px[["Close", "Volume"]], how="left")
    j["ret"] = j["Close"] / j["Close"].shift(1) - 1
    j["mret"] = j["MktClose"] / j["MktClose"].shift(1) - 1
    j["ar"] = j["ret"] - j["mret"]
    dates = j.index.date
    ret, mret, ar, vol = (j[c].to_numpy() for c in ("ret", "mret", "ar", "Volume"))
    rows, dropped = [], []
    for r in events.itertuples():
        pos = int(np.searchsorted(dates, r.report_date))  # first trading day >= report_date
        if pos >= len(j):
            reason = "t0 after last available trading day"
        elif pos - 10 < 0 or pos - 2 < 0:
            reason = "insufficient pre-event history"
        elif pos + 5 >= len(j):
            reason = "t0+5 not yet available"
        elif np.isnan(ret[pos - 1: pos + 6]).any() or np.isnan(mret[pos - 1: pos + 6]).any():
            reason = "missing price in event window"
        elif np.isnan(vol[pos - 10: pos + 1]).any():
            reason = "missing volume in baseline window"
        else:
            reason = None
        if reason:
            dropped.append((r.accession_number, r.ticker, r.report_date, reason))
            continue
        base_vol = vol[pos - 10: pos].mean()
        car1 = ar[pos - 1: pos + 2].sum()
        rows.append({
            "accession_number": r.accession_number,
            "trading_date": dates[pos],
            "daily_return": ret[pos],
            "market_return": mret[pos],
            "abnormal_return": ar[pos],
            "car_minus1_plus1": car1,
            "car_minus1_plus5": ar[pos - 1: pos + 6].sum(),
            "volume_change_pct": (vol[pos] - base_vol) / base_vol if base_vol else np.nan,
            "reaction_category": P1.reaction_category(car1),
        })
    return rows, dropped


def derive_events(filings, ticker):
    """Phase 1 build_events() derivations for one company's 8-K rows."""
    ev = filings[filings["form"] == "8-K"].copy()
    ev["report_date"] = pd.to_datetime(ev["reportDate"].where(ev["reportDate"] != "", ev["filingDate"])).dt.date
    ev["filing_date"] = pd.to_datetime(ev["filingDate"]).dt.date
    items = ev["items"].fillna("").apply(lambda s: [i.strip() for i in s.split(",") if i.strip()])
    ev["item_codes"] = items.apply(",".join)
    ev["event_category"] = items.apply(lambda l: P1.ITEM_TO_CATEGORY.get(P1.primary_item(l), "Other"))
    ev["num_items_disclosed"] = items.apply(len)
    ev["lag_days"] = (pd.to_datetime(ev["filing_date"]) - pd.to_datetime(ev["report_date"])).dt.days
    ev["filing_timeliness"] = ev["lag_days"].apply(P1.timeliness)
    ev["ticker"] = ticker
    return ev.rename(columns={"accessionNumber": "accession_number"})


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def stage_build(limit=None):
    t_start = time.time()
    L = []

    def out(msg=""):
        print(msg, flush=True)
        L.append(msg)

    idx = load_prices("^GSPC")
    last_mkt = idx.index[-1].date()
    cands = candidates(limit)
    out(f"EECS 4412 Phase 2 build: {datetime.now():%Y-%m-%d %H:%M:%S}"
        + (f"  [SMOKE TEST --limit {limit}]" if limit else ""))
    out(f"SEC User-Agent: {P1.SEC_HEADERS['User-Agent']}")
    out(f"Methodology imported from {PHASE1 / 'pipeline.py'} (event_category map, primary_item, "
        f"timeliness, reaction_category); reaction formulas mirror Phase 1 build_reactions().")

    # ---- company funnel --------------------------------------------------
    funnel, recs, kept = [], {}, []
    for c in cands.itertuples():
        row = {"cik": c.cik, "ticker": c.ticker, "title": c.title, "exchange_sec": c.exchange_sec}
        p = CACHE / "sec" / f"{c.cik}.json.gz"
        if not p.exists():
            row["outcome"] = "SEC record not fetched"
            funnel.append(row)
            continue
        rec = read_gz_json(p)
        row.update(sec_status=rec["status"], entity_type=rec.get("entityType"),
                   sic=rec.get("sic"), sic_description=rec.get("sicDescription"))
        if rec["status"] != "ok":
            row["outcome"] = {"not_operating": "SEC entityType not 'operating' (fund/foreign/other)",
                              "blank_check": "SIC 6770 blank-check company (SPAC)",
                              "no_8k_in_window": "no 8-K filed in window",
                              "not_found": "SEC submissions record not found"}[rec["status"]]
            funnel.append(row)
            continue
        px = load_prices(c.ticker)
        if px is None:
            row["outcome"] = "no Yahoo Finance price history"
            funnel.append(row)
            continue
        vol, why = trailing_vol(px, idx)
        row.update(price_first=px.index.min().date(), price_last=px.index.max().date())
        if vol is None:
            row["outcome"] = why
            funnel.append(row)
            continue
        ip = info_path(c.ticker)
        info = json.loads(ip.read_text()) if ip.exists() else {"error": "not fetched"}
        if "error" in info:
            row["outcome"] = f"Yahoo fundamentals unavailable ({info['error'][:60]})"
        elif info.get("quoteType") not in (None, "EQUITY"):
            row["outcome"] = f"Yahoo quoteType {info['quoteType']} (not common equity)"
        elif info.get("industry") == "Shell Companies":
            # second blank-check screen: SPACs often carry their target industry's SIC code
            row["outcome"] = "Yahoo industry 'Shell Companies' (SPAC not coded SIC 6770 at SEC)"
        elif not info.get("marketCap"):
            row["outcome"] = "market cap unavailable (info and Phase 1 fallbacks)"
        else:
            row["outcome"] = "included"
            row["trailing_volatility_1y"] = vol
            kept.append(c.ticker)
            recs[c.ticker] = (rec, info, px)
        funnel.append(row)
    funnel = pd.DataFrame(funnel)

    out("\n== Universe & company funnel ==")
    u = json.loads((CACHE / "stats_universe.json").read_text())
    out(f"  SEC ticker list (reference/company_tickers.json): {u['reference_rows']} tickers")
    for k, v in u["excluded"].items():
        out(f"    excluded: {v:>5}  {k}")
    out(f"  candidate companies (one primary NYSE/Nasdaq common per CIK): {u['candidates']}"
        + (f"  [smoke test uses {len(cands)}]" if limit else ""))
    for k, v in funnel["outcome"].value_counts().items():
        out(f"    {v:>5}  {k}")

    # ---- companies & tiers ------------------------------------------------
    comp = funnel[funnel["outcome"] == "included"].copy()
    comp["company_name"] = [recs[t][0]["name"] for t in comp["ticker"]]
    comp["sector"] = [recs[t][1].get("sector") for t in comp["ticker"]]
    comp["sub_industry"] = [recs[t][1].get("industry") for t in comp["ticker"]]
    comp["exchange"] = [recs[t][1].get("exchange") for t in comp["ticker"]]
    comp["market_cap"] = [recs[t][1]["marketCap"] for t in comp["ticker"]]
    comp["revenue_ttm"] = [recs[t][1].get("totalRevenue") for t in comp["ticker"]]
    comp["employees"] = [recs[t][1].get("fullTimeEmployees") for t in comp["ticker"]]
    comp["market_cap_source"] = [recs[t][1].get("market_cap_source") for t in comp["ticker"]]
    comp["size_category"] = pd.qcut(comp["market_cap"].astype(float), 4, labels=SIZE_LABELS)
    comp["volatility_tier"] = pd.qcut(comp["trailing_volatility_1y"], 4, labels=VOL_LABELS)
    comp["selection_method"] = SELECTION
    cap_edges = pd.qcut(comp["market_cap"].astype(float), 4, retbins=True)[1]
    vol_edges = pd.qcut(comp["trailing_volatility_1y"], 4, retbins=True)[1]

    # ---- events ---------------------------------------------------------------
    ev_frames = []
    for t in comp["ticker"]:
        f = pd.DataFrame(recs[t][0]["filings"])
        if not f.empty:
            ev_frames.append(derive_events(f, t))
    ev = pd.concat(ev_frames, ignore_index=True)
    ev = ev[(ev["report_date"] >= WINDOW_START) & (ev["report_date"] <= date.today())]
    n_dup = int(ev.duplicated("accession_number").sum())
    ev = ev.drop_duplicates("accession_number")  # co-registrant filings listed under two CIKs

    # ---- per-year data quality on the raw filing metadata ------------------
    ev["year"] = [d.year for d in ev["report_date"]]
    yq = ev.groupby("year").agg(n=("accession_number", "size"),
                                tagged=("item_codes", lambda s: (s != "").mean()),
                                has_report_date=("reportDate", lambda s: (s != "").mean()))

    # ---- reactions --------------------------------------------------------------
    rows, dropped = [], []
    for t, sub in ev.groupby("ticker"):
        r, d = reactions_for_ticker(recs[t][2], idx, sub)
        rows += r
        dropped += d
    mr = pd.DataFrame(rows)
    priced = ev["accession_number"].isin(set(mr["accession_number"]))
    yq["priced"] = priced.groupby(ev["year"]).mean()

    # Earliest year from which every later year passes the quality thresholds.
    ok = (yq["tagged"] >= YEAR_MIN_TAGGED) & (yq["has_report_date"] >= YEAR_MIN_TAGGED)
    ok &= (yq["priced"] >= YEAR_MIN_PRICED) | (yq.index == date.today().year)  # current year: t0+5 pending
    start_year = WINDOW_START.year
    for y in sorted(yq.index, reverse=True):
        if not ok[y]:
            start_year = y + 1
            break
    eff_start = date(start_year, 1, 1)

    out("\n== Time range ==")
    out(f"  requested: {WINDOW_START} to {date.today()};  last ^GSPC session: {last_mkt}")
    out(f"  per-year quality (thresholds: items tagged >= {YEAR_MIN_TAGGED:.0%}, reportDate present "
        f">= {YEAR_MIN_TAGGED:.0%}, events with complete price window >= {YEAR_MIN_PRICED:.0%}):")
    out(f"    {'year':<6}{'8-Ks':>8}{'tagged':>9}{'reportDate':>12}{'priced':>9}  pass")
    for y, r in yq.iterrows():
        out(f"    {y:<6}{int(r.n):>8}{r.tagged:>9.1%}{r.has_report_date:>12.1%}{r.priced:>9.1%}  "
            f"{'yes' if ok[y] else 'NO'}")
    if eff_start > WINDOW_START:
        out(f"  -> data quality fails in {start_year - 1}; window starts {eff_start} (not pushed further back)")
    else:
        out(f"  -> every year passes; full requested range achieved: {eff_start} to {date.today()}")

    keep_ev = ev["report_date"] >= eff_start
    cut_accessions = set(ev.loc[~keep_ev, "accession_number"])
    ev = ev[keep_ev]
    mr = mr[~mr["accession_number"].isin(cut_accessions)]
    dropped = [d for d in dropped if d[0] not in cut_accessions]
    ev = ev[ev["accession_number"].isin(set(mr["accession_number"]))]
    ev = ev.sort_values(["ticker", "report_date", "accession_number"]).reset_index(drop=True)
    ev["event_id"] = [f"P2E{i:06d}" for i in range(1, len(ev) + 1)]
    mr = mr.merge(ev[["accession_number", "event_id"]], on="accession_number")
    mr = mr.set_index("event_id").loc[ev["event_id"]].reset_index()

    # companies whose every event was dropped are removed (no orphan rows)
    has_events = set(ev["ticker"])
    no_ev = comp[~comp["ticker"].isin(has_events)]
    comp = comp[comp["ticker"].isin(has_events)]

    comp_out = comp[["ticker", "company_name", "sector", "sub_industry", "exchange", "cik", "sic",
                     "sic_description", "size_category", "market_cap", "revenue_ttm", "employees",
                     "trailing_volatility_1y", "volatility_tier", "selection_method"]].rename(
        columns={"sic": "sic_code"})
    for col in ("market_cap", "revenue_ttm", "employees"):
        comp_out[col] = comp_out[col].astype("Int64")
    ev_out = ev[["event_id", "ticker", "accession_number", "item_codes", "event_category",
                 "filing_date", "report_date", "filing_timeliness", "num_items_disclosed"]]
    mr_out = mr[["event_id", "trading_date", "daily_return", "market_return", "abnormal_return",
                 "car_minus1_plus1", "car_minus1_plus5", "volume_change_pct", "reaction_category"]]

    PROC.mkdir(exist_ok=True)
    comp_out.to_csv(PROC / "companies.csv", index=False, float_format="%.6f")
    ev_out.to_csv(PROC / "events.csv", index=False)
    mr_out.to_csv(PROC / "market_reactions.csv", index=False, float_format="%.6f")
    funnel.to_csv(RAW / "company_funnel.csv", index=False)
    pd.DataFrame(dropped, columns=["accession_number", "ticker", "report_date", "reason"]).to_csv(
        RAW / "dropped_events.csv", index=False)

    # ---- report ---------------------------------------------------------------
    out("\n== Final counts ==")
    out(f"  companies.csv         {len(comp_out):>8}")
    out(f"  events.csv            {len(ev_out):>8}")
    out(f"  market_reactions.csv  {len(mr_out):>8}")
    if len(no_ev):
        out(f"  ({len(no_ev)} otherwise-eligible companies removed: none of their 8-Ks had a complete "
            f"price window)")

    out("\n== Sampling design ==")
    out(f"  selection_method = '{SELECTION}': every company that passed verification is included,")
    out("  i.e. a 100% sampling fraction in every sector x size x volatility stratum.")
    out(f"  plain stratified (census) companies: {len(comp_out)};  purposive top-up: 0 (none needed:")
    out("  tiers are quartiles of the final pool, so no tier can be starved).")

    def table(title, s):
        out(f"\n  {title}")
        for k, v in s.items():
            out(f"    {str(k):<26}{v:>7}  {pct(v, s.sum())}")

    out("\n== Stratification dimensions ==")
    out("  volatility_tier = quartiles of trailing 1-year realized volatility (annualized std of daily")
    out(f"  returns, last {VOL_DAYS} ^GSPC sessions to {last_mkt}, >= {VOL_MIN_OBS} valid returns required).")
    out("  Tier edges (annualized): " + ", ".join(f"{e:.1%}" for e in vol_edges))
    table("volatility_tier (companies)", comp_out["volatility_tier"].value_counts().reindex(VOL_LABELS))
    ev_tier = ev_out.merge(comp_out[["ticker", "volatility_tier"]], on="ticker")["volatility_tier"]
    table("volatility_tier (events)", ev_tier.value_counts().reindex(VOL_LABELS))
    out("  size_category edges (USD): " + ", ".join(f"{e:,.0f}" for e in cap_edges))
    table("size_category (companies)", comp_out["size_category"].value_counts().reindex(SIZE_LABELS))
    out(f"  market cap source: {comp['market_cap_source'].value_counts().to_dict()}")
    table("sector (Yahoo classification; companies)", comp_out["sector"].fillna("(blank)").value_counts())
    out("\n  size_category x volatility_tier (companies):")
    xt = pd.crosstab(comp_out["size_category"], comp_out["volatility_tier"]).reindex(
        index=SIZE_LABELS, columns=VOL_LABELS, fill_value=0)
    out("    " + f"{'':<10}" + "".join(f"{c:>11}" for c in VOL_LABELS))
    for k, r in xt.iterrows():
        out("    " + f"{k:<10}" + "".join(f"{v:>11}" for v in r))
    thin = pd.crosstab(comp_out["sector"], comp_out["volatility_tier"]).stack()
    thin = thin[thin < 10]
    out(f"  sector x volatility_tier cells with < 10 companies: {len(thin)}"
        + (" -- " + ", ".join(f"{s}/{v}={n}" for (s, v), n in thin.items()) if len(thin) else ""))

    p1 = pd.read_csv(PHASE1 / "processed" / "companies.csv")
    p1_tiers = comp_out[comp_out["ticker"].isin(p1["ticker"])]["volatility_tier"].value_counts().reindex(VOL_LABELS)
    out(f"\n  Phase 1 companies on Phase 2's volatility scale ({int(p1_tiers.sum())} of {len(p1)} matched): "
        + ", ".join(f"{k} {v}" for k, v in p1_tiers.items()))

    out("\n== Filing timeliness (not curated; Phase 1 calendar-day buckets) ==")
    table("filing_timeliness (events)", ev_out["filing_timeliness"].value_counts().reindex(
        ["On-time", "Slightly Late", "Late"]).fillna(0).astype(int))
    late = ev_out["filing_timeliness"] != "On-time"
    out(f"  Slightly Late + Late share: {pct(int(late.sum()), len(ev_out))}"
        + ("  -- THIN (< 5%) even in the broad universe" if late.mean() < 0.05 else ""))
    sl = ev[ev["filing_timeliness"] == "Slightly Late"]
    bd = np.busday_count(np.array(sl["report_date"], dtype="datetime64[D]"),
                         np.array(sl["filing_date"], dtype="datetime64[D]"))
    out(f"  caveat: {pct(int((bd <= 4).sum()), len(sl))} of 'Slightly Late' events were filed within 4 "
        f"business days (weekends; holidays not counted), i.e. on time under SEC's rule.")
    out(f"  negative lags (reportDate after filingDate, kept as filed): {int((ev['lag_days'] < 0).sum())}")
    tl = ev_out.merge(comp_out[["ticker", "size_category"]], on="ticker")
    out("  non-On-time share by size_category: " + ", ".join(
        f"{k} {pct(int((g['filing_timeliness'] != 'On-time').sum()), len(g))}"
        for k, g in tl.groupby("size_category", observed=True)))

    table("event_category (events)", ev_out["event_category"].value_counts())
    table("reaction_category (events)", mr_out["reaction_category"].value_counts())

    out("\n== Dropped events ==")
    out(f"  duplicate accession numbers (co-registrant filings) removed: {n_dup}")
    out(f"  events before effective start {eff_start}: {len(cut_accessions)}")
    for k, v in Counter(d[3] for d in dropped).most_common():
        out(f"    {v:>7}  {k}")
    out("  (per-event list: raw/dropped_events.csv; per-company outcomes: raw/company_funnel.csv)")
    out(f"  volume_change_pct blank (zero baseline volume): {int(mr_out['volume_change_pct'].isna().sum())}")

    # ---- methodology check against the frozen Phase 1 data ---------------------
    out("\n== Methodology check vs Phase 1 ==")
    p1e = pd.read_csv(PHASE1 / "processed" / "events.csv", parse_dates=["report_date"])
    p1m = pd.read_csv(PHASE1 / "processed" / "market_reactions.csv")
    p1e["report_date"] = p1e["report_date"].dt.date
    chk = []
    for t, sub in p1e.groupby("ticker"):
        px = load_prices(t)
        if px is not None:
            chk += reactions_for_ticker(px, idx, sub)[0]
    if chk:
        c2 = pd.DataFrame(chk).merge(p1e[["accession_number", "event_id"]], on="accession_number").merge(
            p1m, on="event_id", suffixes=("_p2", "_p1"))
        cols = ["daily_return", "market_return", "abnormal_return", "car_minus1_plus1", "car_minus1_plus5"]
        worst = max(float((c2[f"{k}_p2"] - c2[f"{k}_p1"]).abs().max()) for k in cols)
        same_cat = float((c2["reaction_category_p2"] == c2["reaction_category_p1"]).mean())
        out(f"  recomputed {len(c2)} of {len(p1e)} Phase 1 events with Phase 2 code: max |diff| in returns/"
            f"CARs = {worst:.2e}; reaction_category identical for {same_cat:.1%}")
    else:
        out("  skipped (Phase 1 tickers not in this run's price cache)")

    # ---- data-quality caveats --------------------------------------------------
    x = ev_out.merge(mr_out, on="event_id")
    extreme = x[(x["daily_return"].abs() > EXTREME_MOVE) | (x["car_minus1_plus1"].abs() > EXTREME_MOVE)
                | (x["car_minus1_plus5"].abs() > EXTREME_MOVE)]
    extreme.to_csv(RAW / "extreme_return_events.csv", index=False, float_format="%.6f")
    same_day = ev_out.groupby(["ticker", "report_date"]).size()
    out("\n== Data-quality caveats for modeling ==")
    out(f"  1. Extreme returns kept as reported by Yahoo: {len(extreme)} events have |daily_return| or a CAR "
        f"above {EXTREME_MOVE:.0%}")
    for th in (0.5, 1.0, 3.0):
        out(f"       |daily_return| > {th:.0%}: {int((x['daily_return'].abs() > th).sum()):>6}   "
            f"|CAR[-1,+5]| > {th:.0%}: {int((x['car_minus1_plus5'].abs() > th).sum()):>6}")
    out("     Some are genuine micro-cap moves; some are likely unadjusted reverse splits (e.g. NFE shows a")
    out("     one-day +5,060% close on 2026-08-04, outside any event window). Not removed or winsorized here;")
    out("     listed in raw/extreme_return_events.csv. Use winsorizing/robust methods or a Small-cap sensitivity check.")
    out("  2. Survivorship bias: the universe is TODAY's SEC ticker list, so companies delisted, acquired or")
    out("     bankrupt between 2016 and now are absent. Early years over-represent eventual survivors.")
    out(f"  3. Tiers are measured once, as of {last_mkt} (market cap at fetch, volatility over the trailing year),")
    out("     not at each event date. A 2016 event carries the company's 2026 tier.")
    out("  4. sector/sub_industry are Yahoo's (Morningstar-style) taxonomy, not GICS (Phase 1 used GICS).")
    out("     sic_code/sic_description are SEC's own classification, included alongside.")
    out("  5. filing_timeliness keeps Phase 1's calendar-day buckets; most 'Slightly Late' filings were on time")
    out("     under SEC's 4-business-day rule (see timeliness section). Real late filing is rarer than it looks.")
    out(f"  6. {int((same_day > 1).sum())} ticker/date pairs have more than one 8-K "
        f"({int(same_day[same_day > 1].sum())} events); they share one price reaction, so they are not independent.")
    out("  7. ticker = the company's CURRENT ticker; events are pulled by CIK, so ticker changes are handled,")
    out("     but price history before a ticker change may be missing on Yahoo (those events are dropped above).")
    out("  8. Known residual: FACT (FACT II Acquisition Corp.) is a SPAC that neither screen caught (SEC SIC")
    out("     3728, no Yahoo industry). 8-K/A amendments are excluded, as in Phase 1.")

    out("\n== Effort ==")
    tot_s = 0.0
    for st in ("sec", "prices", "info"):
        p = CACHE / f"stats_{st}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        tot_s += s.get("elapsed_s", 0)
        if st == "prices":
            n_hist = len(list((CACHE / "prices").glob("*.csv.gz")))
            out(f"  prices  {s.get('requests', 0):>7} yfinance download calls (batched; ~{n_hist} ticker "
                f"histories, i.e. roughly that many HTTP requests)  {s.get('elapsed_s', 0) / 3600:5.2f} h")
        else:
            extra = {k: v for k, v in s.items() if k not in ("requests", "elapsed_s", "elapsed_s_prev")}
            host = "data.sec.gov" if st == "sec" else "Yahoo quoteSummary"
            out(f"  {st:<7} {s.get('requests', 0):>7} {host} requests  {s.get('elapsed_s', 0) / 3600:5.2f} h  {extra}")
    out(f"  stage time {tot_s / 3600:.2f} h total (sec and prices ran in parallel, so wall-clock was less)")
    out(f"  SEC rate cap {SEC_RATE:.0f} req/s (SEC limit 10), {SEC_THREADS} threads, back-off on 403/429/5xx")
    out(f"  build   {time.time() - t_start:.0f}s")

    (PROC / "run_log.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["universe", "sec", "prices", "info", "build", "all"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    for d in (RAW, PROC, CACHE / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    stages = ["universe", "sec", "prices", "info", "build"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "universe":
            stage_universe()
        else:
            globals()[f"stage_{s}"](a.limit)


if __name__ == "__main__":
    main()
