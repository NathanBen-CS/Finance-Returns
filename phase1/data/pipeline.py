"""
EECS 4412 Phase 1 data pipeline: corporate events (SEC 8-K) & abnormal stock returns.

Implements phase1_data_pipeline_spec.md. Outputs (in ./output):
    companies.csv, events.csv, market_reactions.csv, run_log.txt

Raw API responses are cached in ./cache so re-runs don't re-hit SEC / Yahoo.
Delete ./cache to force a fresh pull.

Usage:  python pipeline.py
Env:    SEC_USER_AGENT  overrides the SEC User-Agent header.
"""
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT / "output"
SAMPLE_CSV = ROOT / "companies_sample_57.csv"
CONSTITUENTS_CSV = ROOT / "constituents.csv"

SEC_HEADERS = {"User-Agent": os.environ.get(
    "SEC_USER_AGENT", "EECS4412 Student Project nathanbenayguev9@gmail.com")}

DEEP_DIVE = ["AAPL", "MSFT", "JNJ", "UNH", "JPM", "GS", "AMZN", "GOOGL",
             "META", "BA", "PG", "XOM", "NEE", "LIN", "AMT"]
SAMPLE_SIZE = 57
SAMPLE_SEED = 4412

# Yahoo lookup symbols that differ from the S&P constituent symbol. The CSV always
# keeps the ticker exactly as it appears in companies_sample_57.csv.
YF_SYMBOL_OVERRIDE = {"FISV": "FI"}
YF_OVERRIDE_REASON = {"FISV": "Fiserv rebranded FISV -> FI in July 2023"}

# ExxonMobil reorganized under a new holding-company registrant in mid-2026.
# The ticker's filing history before that lives under the old Exxon Mobil Corp CIK.
EXTRA_CIKS = {"XOM": [34088]}

WINDOW_DEFAULT = (date(2025, 9, 1), date(2026, 9, 17))
WINDOW_WIDE = (date(2025, 1, 1), date(2026, 9, 17))
MIN_EVENTS, MAX_EVENTS = 60, 300

# Revised mapping (v2). Changes vs the original spec mapping, with rationale:
#   5.07 shareholder votes      Other -> Management/Governance (a governance action)
#   3.03 modification of rights Other -> Legal/Regulatory (same SEC section as 3.01)
#   2.05 exit/disposal costs    Other -> Financial/Earnings (financial charge, like 2.06)
#   7.01 Reg FD disclosure      Capital/Financing -> Financial/Earnings (guidance /
#                               preliminary results, not financing)
# Capital/Financing is then just 2.03 (new debt/credit obligations); Other is 8.01,
# which is SEC's own catch-all item, plus anything unlisted.
CATEGORY_MAP = {
    "Financial/Earnings": ["2.02", "2.06", "1.03", "2.05", "7.01"],
    "M&A/Contracts": ["1.01", "1.02", "2.01", "5.01"],
    "Management/Governance": ["5.02", "4.01", "5.03", "5.07"],
    "Legal/Regulatory": ["3.01", "4.02", "3.03"],
    "Capital/Financing": ["2.03"],
    # Item 1.05 (Material Cybersecurity Incidents, required since Dec 2023) was
    # previously unmapped and fell into Other; added with the cyber-incident extension.
    "Security Incident": ["1.05"],
    "Other": ["8.01"],
}
ITEM_TO_CATEGORY = {code: cat for cat, codes in CATEGORY_MAP.items() for code in codes}

# Kept only to log the before/after shift for the write-up.
CATEGORY_MAP_V1 = {
    "Financial/Earnings": ["2.02", "2.06", "1.03"],
    "M&A/Contracts": ["1.01", "1.02", "2.01", "5.01"],
    "Management/Governance": ["5.02", "4.01", "5.03"],
    "Legal/Regulatory": ["3.01", "4.02"],
    "Capital/Financing": ["2.03", "7.01"],
    "Other": ["8.01", "2.05"],
}
ITEM_TO_CATEGORY_V1 = {c: cat for cat, codes in CATEGORY_MAP_V1.items() for c in codes}

# Events dropped by the 2026-09-16 run because t0+5 had not traded yet; recoverable now.
PREVIOUSLY_DROPPED = {
    "0001104659-26-107526": "AMZN 2026-09-14",
    "0001104659-26-107537": "NEE 2026-09-14",
    "0001193125-26-390605": "AMT 2026-09-14",
}

LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)


# ---------------------------------------------------------------------------
# Step 0: 57-company stratified sample
# ---------------------------------------------------------------------------
def load_or_build_sample():
    if SAMPLE_CSV.exists():
        sample = pd.read_csv(SAMPLE_CSV, dtype={"CIK": str, "Founded": str})
        log(f"[sample] Using existing {SAMPLE_CSV.name} ({len(sample)} rows).")
        return sample

    log(f"[sample] {SAMPLE_CSV.name} not found -> building it from {CONSTITUENTS_CSV.name}.")
    cons = pd.read_csv(CONSTITUENTS_CSV, dtype={"CIK": str, "Founded": str})
    missing = set(DEEP_DIVE) - set(cons["Symbol"])
    if missing:
        sys.exit(f"Deep-dive tickers missing from constituents: {missing}")

    # Proportional allocation per GICS sector, largest-remainder rounding to 57.
    counts = cons["GICS Sector"].value_counts()
    exact = counts / counts.sum() * SAMPLE_SIZE
    quota = np.floor(exact).astype(int)
    for sector in (exact - quota).sort_values(ascending=False).index[: SAMPLE_SIZE - quota.sum()]:
        quota[sector] += 1

    rng = np.random.default_rng(SAMPLE_SEED)
    picks = []
    for sector, q in quota.items():
        pool = cons[cons["GICS Sector"] == sector]
        forced = pool[pool["Symbol"].isin(DEEP_DIVE)]
        rest = pool[~pool["Symbol"].isin(DEEP_DIVE)]
        n_rand = q - len(forced)
        if n_rand < 0:
            sys.exit(f"Sector {sector} quota {q} < deep-dive count {len(forced)}")
        idx = rng.choice(len(rest), size=n_rand, replace=False)
        picks.append(forced)
        picks.append(rest.iloc[sorted(idx)])
        log(f"    {sector:<24} population={counts[sector]:>3} quota={q:>2} "
            f"(deep-dive {len(forced)} + random {n_rand})")

    sample = pd.concat(picks).sort_values(["GICS Sector", "Symbol"]).reset_index(drop=True)
    sample["DeepDive"] = np.where(sample["Symbol"].isin(DEEP_DIVE), "Y", "N")
    sample.to_csv(SAMPLE_CSV, index=False)
    log(f"[sample] Wrote {SAMPLE_CSV.name}: {len(sample)} rows, "
        f"{(sample['DeepDive'] == 'Y').sum()} deep-dive (seed={SAMPLE_SEED}).")
    return sample


# ---------------------------------------------------------------------------
# Step 1: companies.csv
# ---------------------------------------------------------------------------
def fetch_info(symbol, retries=3):
    cache_file = CACHE / "yf_info" / f"{symbol}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    fields = ("exchange", "marketCap", "totalRevenue", "fullTimeEmployees")
    # Try the remapped symbol first (if any), then the sample's own symbol. Yahoo serves
    # different field subsets per call, so merge the non-null fields across attempts.
    candidates = [YF_SYMBOL_OVERRIDE[symbol]] if symbol in YF_SYMBOL_OVERRIDE else []
    candidates.append(symbol)
    if len(candidates) > 1:
        log(f"    [yf info] {symbol}: trying Yahoo symbol '{candidates[0]}' first "
            f"({YF_OVERRIDE_REASON.get(symbol, 'symbol remap')}), falling back to "
            f"'{symbol}'; CSV keeps ticker '{symbol}'")

    keep = {k: None for k in fields}
    for cand in candidates:
        yf_symbol = cand.replace(".", "-")  # e.g. BRK.B -> BRK-B
        for attempt in range(1, retries + 1):
            try:
                tk = yf.Ticker(yf_symbol)
                info = tk.get_info()
                if not info or "marketCap" not in info and "exchange" not in info:
                    raise ValueError("empty info payload")
                for k in fields:
                    if keep[k] is None and info.get(k) is not None:
                        keep[k] = info[k]
                if keep["marketCap"] is None:
                    keep["marketCap"] = derive_market_cap(symbol, tk, info)
                if cand != symbol:
                    log(f"    [yf info] {symbol}: resolved via '{yf_symbol}'")
                if all(keep[k] is not None for k in fields):
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(keep))
                    time.sleep(0.7)
                    return keep
                break  # payload arrived but incomplete; try next candidate symbol
            except Exception as e:
                wait = 2 ** attempt
                log(f"    [yf info] {symbol} as '{yf_symbol}' attempt {attempt} failed ({e});"
                    f" sleeping {wait}s")
                time.sleep(wait)

    if all(keep[k] is None for k in fields):
        return None
    missing = [k for k in fields if keep[k] is None]
    log(f"    [yf info] {symbol}: partial payload, Yahoo omitted {missing}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(keep))
    return keep


def load_prior_fundamentals():
    """Real values from an earlier run's Yahoo lookup, used only to fill fields Yahoo
    no longer returns. Never synthesized - each fill is logged with its fetch date."""
    path = CACHE / "prior_fundamentals.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).set_index("ticker")
    return df.to_dict("index")


def derive_market_cap(symbol, ticker_obj, info):
    """Fallback when .info['marketCap'] is empty: shares x price, else fast_info."""
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    if shares and price:
        mc = int(round(shares * price))
        log(f"    [market_cap] {symbol}: .info['marketCap'] empty -> derived from "
            f"sharesOutstanding ({shares:,}) x price ({price}) = {mc:,}")
        return mc
    try:
        fi = ticker_obj.fast_info
        mc = fi.get("market_cap") if hasattr(fi, "get") else getattr(fi, "market_cap", None)
        if not mc:
            shares = (fi.get("shares") if hasattr(fi, "get") else getattr(fi, "shares", None))
            price = (fi.get("last_price") if hasattr(fi, "get") else getattr(fi, "last_price", None))
            mc = shares * price if shares and price else None
        if mc:
            log(f"    [market_cap] {symbol}: derived from fast_info = {int(mc):,}")
            return int(round(mc))
    except Exception as e:
        log(f"    [market_cap] {symbol}: fast_info fallback failed ({e})")
    log(f"    [market_cap] {symbol}: no market cap available from Yahoo -> left blank")
    return None


def first_year(value):
    m = re.search(r"\d{4}", str(value))
    return int(m.group()) if m else pd.NA


def build_companies(sample):
    log("\n[companies] Fetching fundamentals from yfinance ...")
    rows, failed = [], []
    prior = load_prior_fundamentals()
    prior_field = {"exchange": "exchange", "marketCap": "market_cap",
                   "totalRevenue": "revenue_ttm", "fullTimeEmployees": "employees"}
    for _, r in sample.iterrows():
        sym = r["Symbol"]
        info = fetch_info(sym)
        if info is None:
            failed.append(sym)
            info = {}
        for field in ("exchange", "marketCap", "totalRevenue", "fullTimeEmployees"):
            if info.get(field) is not None:
                continue
            old = prior.get(sym, {})
            val = old.get(prior_field[field])
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                info[field] = int(val) if field != "exchange" else val
                log(f"    [companies] {sym}: '{field}' missing from Yahoo now -> kept real value"
                    f" {info[field]!r} from the {old.get('fetched')} lookup")
            else:
                log(f"    [companies] {sym}: '{field}' not provided by Yahoo -> left blank")
        rows.append({
            "ticker": r["Symbol"],
            "company_name": r["Security"],
            "sector": r["GICS Sector"],
            "sub_industry": r["GICS Sub-Industry"],
            "exchange": info.get("exchange"),
            "cik": int(r["CIK"]),
            "founded_year": first_year(r["Founded"]),
            "market_cap": info.get("marketCap"),
            "revenue_ttm": info.get("totalRevenue"),
            "employees": info.get("fullTimeEmployees"),
        })
    df = pd.DataFrame(rows)
    for c in ("market_cap", "revenue_ttm", "employees"):
        df[c] = df[c].astype("Int64")
    df["founded_year"] = df["founded_year"].astype("Int64")

    cleaned = sample[sample["Founded"].astype(str).str.fullmatch(r"\d{4}") == False]
    for _, r in cleaned.iterrows():
        log(f"    [founded_year] {r['Symbol']}: '{r['Founded']}' -> {first_year(r['Founded'])}")

    mc = df["market_cap"].astype("float")
    df["size_category"] = pd.qcut(mc, 4, labels=["Small", "Mid", "Large", "Mega-cap"])
    edges = pd.qcut(mc, 4, retbins=True)[1]
    log("    [size_category] quartile edges (USD): " + ", ".join(f"{e:,.0f}" for e in edges))

    df = df[["ticker", "company_name", "sector", "sub_industry", "exchange", "cik",
             "founded_year", "size_category", "market_cap", "revenue_ttm", "employees"]]
    if failed:
        log(f"    [companies] yfinance failed after retries for: {failed} (rows kept, fundamentals blank)")
    return df, failed


# ---------------------------------------------------------------------------
# Step 2: events.csv
# ---------------------------------------------------------------------------
def sec_get_json(url, cache_name):
    cache_file = CACHE / "sec" / cache_name
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    for attempt in range(1, 4):
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        time.sleep(0.2)
        if resp.status_code == 200:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(resp.text)
            return resp.json()
        log(f"    [sec] {url} -> HTTP {resp.status_code} (attempt {attempt})")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"SEC request failed: {url}")


def filings_frame(block):
    cols = ["form", "filingDate", "reportDate", "items", "accessionNumber", "primaryDocument"]
    return pd.DataFrame({c: block.get(c, []) for c in cols})


def fetch_8k_filings(cik, earliest_needed):
    """All 8-K rows for a CIK, pulling older paginated files if `recent` doesn't reach back far enough."""
    sub = sec_get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"CIK{cik:010d}.json")
    frames = [filings_frame(sub["filings"]["recent"])]
    recent_oldest = min(frames[0]["filingDate"]) if len(frames[0]) else "9999"
    if recent_oldest > earliest_needed.isoformat():
        for f in sub["filings"].get("files", []):
            if f["filingTo"] >= earliest_needed.isoformat():
                page = sec_get_json(f"https://data.sec.gov/submissions/{f['name']}", f["name"])
                frames.append(filings_frame(page))
                log(f"    [sec] CIK {cik}: recent only reaches {recent_oldest}; "
                    f"also loaded {f['name']} ({f['filingFrom']}..{f['filingTo']})")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"] == "8-K"].drop_duplicates("accessionNumber").copy()
    df["source_cik"] = cik
    return df


def primary_item(items):
    # 9.01 (Financial Statements & Exhibits) is an attachment item, never the event itself;
    # skip it when choosing the primary item unless it's the only one.
    substantive = [i for i in items if i != "9.01"]
    return (substantive or items or [""])[0]


def timeliness(days):
    if days <= 4:
        return "On-time"
    if days <= 10:
        return "Slightly Late"
    return "Late"


def collect_raw_events(companies):
    earliest = WINDOW_WIDE[0]  # fetch enough history for either window
    frames = []
    for t in DEEP_DIVE:
        ciks = [int(companies.loc[companies["ticker"] == t, "cik"].iloc[0])] + EXTRA_CIKS.get(t, [])
        for cik in ciks:
            df = fetch_8k_filings(cik, earliest)
            df["ticker"] = t
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["report_date"] = raw["reportDate"].where(raw["reportDate"].astype(bool), raw["filingDate"])
    raw["report_date"] = pd.to_datetime(raw["report_date"]).dt.date
    raw["filing_date"] = pd.to_datetime(raw["filingDate"]).dt.date
    return raw


def select_window(raw):
    def in_window(w):
        return raw[(raw["report_date"] >= w[0]) & (raw["report_date"] <= w[1])]

    window, reason = WINDOW_DEFAULT, "default"
    ev = in_window(window)
    if len(ev) < MIN_EVENTS:
        window, reason = WINDOW_WIDE, f"widened (default gave {len(ev)} < {MIN_EVENTS})"
        ev = in_window(window)
    elif len(ev) > MAX_EVENTS:
        end = WINDOW_DEFAULT[1]
        window = ((pd.Timestamp(end) - pd.DateOffset(months=8)).date(), end)
        reason = f"narrowed to 8 months (default gave {len(ev)} > {MAX_EVENTS})"
        ev = in_window(window)
    log(f"\n[events] Window used: {window[0]} to {window[1]} ({reason}); {len(ev)} 8-K filings in window")
    return ev.copy(), window


def build_events(raw_window):
    ev = raw_window.sort_values(["ticker", "report_date", "accessionNumber"]).reset_index(drop=True)
    item_lists = ev["items"].fillna("").apply(lambda s: [i.strip() for i in s.split(",") if i.strip()])
    ev["item_codes"] = item_lists.apply(",".join)
    ev["event_category"] = item_lists.apply(lambda l: ITEM_TO_CATEGORY.get(primary_item(l), "Other"))
    ev["num_items_disclosed"] = item_lists.apply(len)
    lag = (pd.to_datetime(ev["filing_date"]) - pd.to_datetime(ev["report_date"])).dt.days
    ev["filing_timeliness"] = lag.apply(timeliness)
    ev["lag_days"] = lag
    ev = ev.rename(columns={"accessionNumber": "accession_number"})

    log("    per-ticker 8-K counts in window:")
    counts = ev["ticker"].value_counts().reindex(DEEP_DIVE, fill_value=0)
    for t, n in counts.items():
        flag = "   <-- few filings" if n < 4 else ""
        src = ev.loc[ev["ticker"] == t, "source_cik"].value_counts().to_dict()
        log(f"      {t:<6} {n:>3}  (by CIK: {src}){flag}")
    unmapped = sorted({primary_item(l) for l in item_lists} - set(ITEM_TO_CATEGORY))
    if unmapped:
        log(f"    primary item codes not in mapping (-> 'Other'): {unmapped}")

    # Before/after comparison of the revised category mapping (same item_codes, re-derived).
    v1 = item_lists.apply(lambda l: ITEM_TO_CATEGORY_V1.get(primary_item(l), "Other"))
    cats = sorted(set(v1) | set(ev["event_category"]))
    log("    [category mapping] counts under original spec map (v1) vs revised map (v2),"
        " over all in-window filings:")
    for c in cats:
        n1, n2 = int((v1 == c).sum()), int((ev["event_category"] == c).sum())
        log(f"      {c:<24} v1={n1:>3}  v2={n2:>3}  ({n2 - n1:+d})")
    moved = ev.loc[v1 != ev["event_category"]]
    log(f"    [category mapping] {len(moved)} filings changed category; by primary item:")
    for item, grp in moved.groupby(moved["item_codes"].apply(lambda s: primary_item(s.split(",")))):
        log(f"      item {item}: {len(grp)} filings  "
            f"{v1.loc[grp.index].iloc[0]} -> {grp['event_category'].iloc[0]}")
    return ev


# ---------------------------------------------------------------------------
# Step 3: market_reactions.csv
# ---------------------------------------------------------------------------
def download_prices(symbol, start, end, need_through=None):
    """Daily prices for `symbol`. A cached file is reused only if it spans `start`
    through `need_through` (the last date this symbol's events actually require), so
    stale caches are re-fetched for tickers with recent events and left alone otherwise."""
    slug = symbol.replace("^", "IDX_")
    cache_dir = CACHE / "prices"
    cache_file = cache_dir / f"{slug}.csv"
    need = pd.Timestamp(need_through or end)

    candidates = [cache_file] + sorted(cache_dir.glob(f"{slug}_*.csv"), reverse=True)
    for cand in candidates:
        if not cand.exists():
            continue
        df = pd.read_csv(cand, index_col=0, parse_dates=True)
        if df.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=5) and df.index.max() >= need:
            if cand != cache_file:
                df.to_csv(cache_file)  # migrate legacy date-stamped cache name
            log(f"    [prices] {symbol}: reusing cache ({df.index.min().date()}..{df.index.max().date()},"
                f" covers required {need.date()})")
            return df
        log(f"    [prices] {symbol}: cache {cand.name} ends {df.index.max().date()}"
            f" < required {need.date()} -> re-fetching")
        break

    for attempt in range(1, 4):
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("no rows returned")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Close", "Volume"]].dropna(subset=["Close"])
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file)
            time.sleep(0.5)
            return df
        except Exception as e:
            log(f"    [prices] {symbol} attempt {attempt} failed ({e})")
            time.sleep(2 ** attempt)
    return None


def reaction_category(car):
    if car < -0.03:
        return "Strong Negative"
    if car < -0.005:
        return "Negative"
    if car <= 0.005:
        return "Neutral"
    if car <= 0.03:
        return "Positive"
    return "Strong Positive"


def build_reactions(events):
    start = (min(events["report_date"]) - timedelta(days=45)).isoformat()   # >=15 + 10 vol-days of trading
    end = min(max(events["report_date"]) + timedelta(days=30), date.today() + timedelta(days=1)).isoformat()
    log(f"\n[prices] Daily prices {start} .. {end} (end exclusive)")

    # Last calendar date any event needs: its t0 plus 5 trading days (<=9 calendar days
    # with a weekend and a holiday), capped at the requested end.
    def coverage(report_dates):
        return min(max(report_dates) + timedelta(days=9), pd.Timestamp(end).date() - timedelta(days=1))

    idx = download_prices("^GSPC", start, end, coverage(events["report_date"]))
    if idx is None:
        sys.exit("Could not download ^GSPC; aborting.")
    last_market_day = idx.index.max().date()
    log(f"    ^GSPC: {len(idx)} trading days, last available {last_market_day}")

    rows, dropped = [], []
    for t in DEEP_DIVE:
        sub = events[events["ticker"] == t]
        px = download_prices(t, start, end, coverage(sub["report_date"]))
        if px is None:
            dropped += [(r.accession_number, t, r.report_date, "no price data for ticker") for r in sub.itertuples()]
            continue
        j = px[["Close", "Volume"]].join(idx[["Close"]].rename(columns={"Close": "MktClose"}), how="inner")
        j["ret"] = j["Close"].pct_change()
        j["mret"] = j["MktClose"].pct_change()
        j["ar"] = j["ret"] - j["mret"]
        dates = j.index.date

        for r in sub.itertuples():
            pos = int(np.searchsorted(dates, r.report_date))  # first trading day >= report_date
            if pos >= len(j):
                reason = "t0 is after last available trading day"
            elif pos - 10 < 0 or pos - 2 < 0:
                reason = "insufficient pre-event history"
            elif pos + 5 >= len(j):
                reason = f"t0+5 not yet available (t0={dates[pos]}, data ends {dates[-1]})"
            elif j.iloc[pos - 1: pos + 6][["ret", "mret"]].isna().any().any():
                reason = "missing price in event window"
            else:
                reason = None
            if reason:
                dropped.append((r.accession_number, t, r.report_date, reason))
                continue
            base_vol = j["Volume"].iloc[pos - 10: pos].mean()
            car1 = j["ar"].iloc[pos - 1: pos + 2].sum()
            rows.append({
                "accession_number": r.accession_number,
                "trading_date": dates[pos],
                "daily_return": j["ret"].iloc[pos],
                "market_return": j["mret"].iloc[pos],
                "abnormal_return": j["ar"].iloc[pos],
                "car_minus1_plus1": car1,
                "car_minus1_plus5": j["ar"].iloc[pos - 1: pos + 6].sum(),
                "volume_change_pct": (j["Volume"].iloc[pos] - base_vol) / base_vol if base_vol else np.nan,
                "reaction_category": reaction_category(car1),
            })
    return pd.DataFrame(rows), dropped


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    log(f"EECS 4412 Phase 1 pipeline run: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"SEC User-Agent: {SEC_HEADERS['User-Agent']}")

    sample = load_or_build_sample()
    companies, failed_info = build_companies(sample)

    raw = collect_raw_events(companies)
    in_window, window = select_window(raw)
    events = build_events(in_window)

    reactions, dropped = build_reactions(events)
    if dropped:
        log(f"\n[drop] {len(dropped)} events dropped (no fabricated values):")
        for acc, t, d, why in dropped:
            log(f"    {t:<6} {d}  {acc}  -- {why}")

    kept = set(reactions["accession_number"])
    log("\n[recovery] Events dropped by the 2026-09-16 run because the t0+5 trading window"
        " had not completed yet (t0 = 2026-09-14, data then ended 2026-09-16).")
    log(f"           Re-fetched today ({date.today()}): t0+5 has now traded, so full"
        " car_minus1_plus1 / car_minus1_plus5 are computable from real prices.")
    for acc, who in PREVIOUSLY_DROPPED.items():
        if acc in kept:
            r = reactions.loc[reactions["accession_number"] == acc].iloc[0]
            log(f"    RECOVERED {who}  {acc}  t0={r['trading_date']} "
                f"CAR[-1,+1]={r['car_minus1_plus1']:+.4f} CAR[-1,+5]={r['car_minus1_plus5']:+.4f}")
        else:
            log(f"    STILL MISSING {who}  {acc} -- see drop list above")

    events = events[events["accession_number"].isin(kept)].reset_index(drop=True)
    events["event_id"] = [f"EVT{i:04d}" for i in range(1, len(events) + 1)]
    reactions = reactions.merge(events[["accession_number", "event_id"]], on="accession_number")
    reactions = reactions.set_index("event_id").loc[events["event_id"]].reset_index()

    events_out = events[["event_id", "ticker", "accession_number", "item_codes", "event_category",
                         "filing_date", "report_date", "filing_timeliness", "num_items_disclosed"]]
    reactions_out = reactions[["event_id", "trading_date", "daily_return", "market_return", "abnormal_return",
                               "car_minus1_plus1", "car_minus1_plus5", "volume_change_pct", "reaction_category"]]

    companies.to_csv(OUT / "companies.csv", index=False)
    events_out.to_csv(OUT / "events.csv", index=False)
    reactions_out.to_csv(OUT / "market_reactions.csv", index=False, float_format="%.6f")

    log("\n[summary]")
    log(f"    companies.csv        {len(companies)} rows ({len(companies.columns)} cols)")
    log(f"    events.csv           {len(events_out)} rows ({len(events_out.columns)} cols)")
    log(f"    market_reactions.csv {len(reactions_out)} rows ({len(reactions_out.columns)} cols)")
    log(f"    event window         {window[0]} to {window[1]}")
    log(f"    yfinance info failures: {failed_info or 'none'}")
    log(f"    events dropped for price reasons: {len(dropped)}")
    log("    event_category counts: " + events_out["event_category"].value_counts().to_dict().__repr__())
    log("    filing_timeliness counts: " + events_out["filing_timeliness"].value_counts().to_dict().__repr__())
    log("    reaction_category counts: " + reactions_out["reaction_category"].value_counts().to_dict().__repr__())
    late = events[events["lag_days"] > 4]
    for r in late.itertuples():
        log(f"    late filing: {r.ticker} report {r.report_date} filed {r.filing_date} "
            f"({r.lag_days} days) items={r.item_codes}")
    dup = events.groupby(["ticker", "report_date"]).size()
    dup = dup[dup > 1]
    if len(dup):
        log(f"    note: {len(dup)} ticker/date pairs have >1 filing (share the same price reaction): "
            + ", ".join(f"{t} {d} x{n}" for (t, d), n in dup.items()))
    log(f"    runtime: {time.time() - t_start:.1f}s")

    (OUT / "run_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
