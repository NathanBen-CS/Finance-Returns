"""
EECS 4412 Phase 1 -- dataset extension: 4 purposively selected companies that
disclosed a material cybersecurity incident (8-K Item 1.05) inside the event window.

Appends to ./processed/companies.csv, events.csv, market_reactions.csv and
run_log.txt. Every fetch and formula is pipeline.py's own function (SEC 8-K
pull, item -> category mapping, timeliness, yfinance fundamentals, the
market-adjusted event study against ^GSPC) -- nothing is reimplemented here.

Existing rows and event_ids are never modified: new events continue numbering
after the current last event_id, and the script refuses to run once the 4
tickers are already present.

Run with: python extend_cyber.py   (from this folder, i.e. data/)
"""
import re
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import pipeline as P

PROC = Path("processed")
CONSTITUENTS = Path("../reference/constituents.csv")
P.CACHE = Path(tempfile.gettempdir()) / "eecs4412_cache"  # keep API cache out of the project

# Hand-verified by the project author against SEC's filing index; re-confirmed
# below against the submissions API before anything is written.
CYBER = [
    # ticker, cik, report_date, accession_number of the Item 1.05 filing
    ("FFIV", 1048695, "2025-10-15", "0001048695-25-000149"),
    ("CPNG", 1834584, "2025-12-15", "0001834584-25-000196"),
    ("AMGN", 318154, "2026-07-29", "0000318154-26-000119"),
    ("BSX", 885725, "2026-09-07", "0000885725-26-000059"),
]
NEW_TICKERS = [c[0] for c in CYBER]

# CPNG is not an S&P 500 constituent, so constituents.csv has no GICS row for it.
# It borrows the GICS label of the dataset company Yahoo classifies identically
# (checked at runtime; left blank if the classifications ever diverge).
GICS_PEER = {"CPNG": "AMZN"}

# Same window the original 15 deep-dive companies were pulled with, so every
# company in events.csv has identical coverage.
WINDOW = P.WINDOW_DEFAULT

SELECTION_EXISTING = "random_stratified"
SELECTION_NEW = "purposive_cyber_incident"
SIZE_LABELS = ["Small", "Mid", "Large", "Mega-cap"]


def read_text(path):
    return path.read_bytes().decode("utf-8")


def category_counts(series):
    return {k: int(v) for k, v in series.value_counts().items()}


def classification_for_non_constituent(symbol, existing):
    """GICS fields for a company outside the S&P 500 list: adopt the GICS label of its
    peer in the dataset, but only if Yahoo classifies the two identically. Founded year
    from Yahoo's company profile."""
    out = {"sector": None, "sub_industry": None, "founded": None}
    peer = GICS_PEER[symbol]
    info = yf.Ticker(symbol).get_info()
    pinfo = yf.Ticker(peer).get_info()
    sec, ind = info.get("sector"), info.get("industry")
    P.log(f"    [classification] {symbol} is not an S&P 500 constituent -> no GICS row in "
          f"constituents.csv. Yahoo: sector='{sec}', industry='{ind}'")
    if sec and (pinfo.get("sector"), pinfo.get("industry")) == (sec, ind):
        row = existing.loc[existing["ticker"] == peer].iloc[0]
        out["sector"], out["sub_industry"] = row["sector"], row["sub_industry"]
        P.log(f"    [classification] {symbol}: Yahoo classification identical to {peer} "
              f"('{sec}' / '{ind}'), whose GICS label here is '{row['sector']}' / "
              f"'{row['sub_industry']}' -> {symbol} takes the same GICS pair")
    else:
        P.log(f"    [classification] {symbol}: Yahoo classification differs from {peer} "
              f"({pinfo.get('sector')} / {pinfo.get('industry')}) -> sector/sub_industry left blank")
    summary = info.get("longBusinessSummary") or ""
    m = re.search(r"(?:founded|incorporated) in (\d{4})", summary)
    if m:
        out["founded"] = m.group(1)
        sentence = re.search(r"[^.]*(?:founded|incorporated) in \d{4}[^.]*\.", summary)
        P.log(f"    [classification] {symbol}: founded_year {m.group(1)} from Yahoo profile: "
              f"\"{sentence.group().strip() if sentence else m.group()}\"")
    else:
        P.log(f"    [classification] {symbol}: no founding year in Yahoo profile -> left blank")
    time.sleep(0.7)
    return out


def main():
    t_start = time.time()
    P.log("")
    P.log("=" * 78)
    P.log(f"EXTENSION RUN {datetime.now():%Y-%m-%d %H:%M:%S} -- extend_cyber.py")
    P.log("Purposive addition of 4 companies that disclosed a material cybersecurity")
    P.log("incident (8-K Item 1.05) inside the event window. Existing rows untouched.")
    P.log("=" * 78)

    comp_path, ev_path, mr_path = PROC / "companies.csv", PROC / "events.csv", PROC / "market_reactions.csv"
    comp_txt, ev_txt, mr_txt = read_text(comp_path), read_text(ev_path), read_text(mr_path)
    comp = pd.read_csv(comp_path, dtype=str, keep_default_na=False)
    ev_old = pd.read_csv(ev_path)
    mr_old = pd.read_csv(mr_path)

    if set(NEW_TICKERS) & set(comp["ticker"]) or "selection_method" in comp.columns:
        sys.exit("Extension already applied (tickers or selection_method present); refusing to append twice.")

    before = {"companies": len(comp), "events": len(ev_old), "market_reactions": len(mr_old)}
    cat_before = category_counts(ev_old["event_category"])

    # --- Events: full 8-K history per company, same path as the original 15 -------
    P.DEEP_DIVE = NEW_TICKERS  # pipeline's event/price loops iterate over DEEP_DIVE
    ciks = pd.DataFrame({"ticker": NEW_TICKERS, "cik": [c[1] for c in CYBER]})
    P.log(f"\n[events] Pulling full 8-K history for {NEW_TICKERS} from SEC EDGAR")
    raw = P.collect_raw_events(ciks)

    # Re-confirm each hand-verified Item 1.05 filing before using it.
    for t, cik, rd, acc in CYBER:
        hit = raw[raw["accessionNumber"] == acc]
        ok = (len(hit) == 1 and "1.05" in hit["items"].iloc[0].split(",")
              and str(hit["report_date"].iloc[0]) == rd and int(hit["source_cik"].iloc[0]) == cik)
        P.log(f"    [verify] {t} {acc}: "
              + (f"present, items={hit['items'].iloc[0]}, report_date={rd} -> OK" if ok
                 else "NOT confirmed against SEC data"))
        if not ok:
            sys.exit(f"Claimed Item 1.05 filing for {t} not confirmed; nothing written.")

    in_win = raw[(raw["report_date"] >= WINDOW[0]) & (raw["report_date"] <= WINDOW[1])].copy()
    after_win = raw[(raw["report_date"] > WINDOW[1])]
    P.log(f"    window: {WINDOW[0]} to {WINDOW[1]} (identical to the original 15 companies)")
    for r in after_win.sort_values("report_date").itertuples():
        P.log(f"    excluded (after window end): {r.ticker} report {r.report_date} "
              f"{r.accessionNumber} items={r.items}")
    P.log("    8-K/A amendments excluded by pipeline.fetch_8k_filings (form == '8-K' only), "
          "same rule as the original 15")

    P.log("    [scope note] the v1/v2 mapping comparison below covers only these new filings:")
    ev_new = P.build_events(in_win)

    # --- Market reactions: pipeline's event study, ^GSPC benchmark -----------------
    reactions, dropped = P.build_reactions(ev_new)
    for acc, t, d, why in dropped:
        P.log(f"    [drop] {t} {d} {acc} -- {why}")
    kept = set(reactions["accession_number"])
    ev_new = ev_new[ev_new["accession_number"].isin(kept)].reset_index(drop=True)

    last_id = int(ev_old["event_id"].str[3:].astype(int).max())
    ev_new["event_id"] = [f"EVT{last_id + i:04d}" for i in range(1, len(ev_new) + 1)]
    reactions = reactions.merge(ev_new[["accession_number", "event_id"]], on="accession_number")
    reactions = reactions.set_index("event_id").loc[ev_new["event_id"]].reset_index()

    ev_cols = list(ev_old.columns)
    mr_cols = list(mr_old.columns)
    ev_out, mr_out = ev_new[ev_cols], reactions[mr_cols]

    # --- Companies: same yfinance fetch as the original 57 ------------------------
    P.log(f"\n[companies] Fetching fundamentals for {NEW_TICKERS} (pipeline.fetch_info)")
    cons = pd.read_csv(CONSTITUENTS, dtype={"CIK": str, "Founded": str}).set_index("Symbol")
    existing_typed = pd.read_csv(comp_path)
    mc57 = existing_typed["market_cap"].astype(float)
    edges = pd.qcut(mc57, 4, retbins=True)[1]
    cut_edges = [-np.inf, *edges[1:-1], np.inf]
    P.log("    size_category: new companies binned on the existing 57-company quartile edges "
          "(USD) " + ", ".join(f"{e:,.0f}" for e in edges[1:-1])
          + " -- the random sample defines the bins; purposive additions don't shift them")

    rows = []
    for t, cik, _, _ in CYBER:
        if t in cons.index:
            c = cons.loc[t]
            name, sector, sub, founded = c["Security"], c["GICS Sector"], c["GICS Sub-Industry"], c["Founded"]
        else:
            sub_info = classification_for_non_constituent(t, existing_typed)
            sec_json = P.sec_get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                                      f"CIK{cik:010d}.json")
            name = sec_json["name"]
            P.log(f"    [classification] {t}: company_name '{name}' from SEC registrant record")
            sector, sub, founded = sub_info["sector"], sub_info["sub_industry"], sub_info["founded"]
        info = P.fetch_info(t) or {}
        for field in ("exchange", "marketCap", "totalRevenue", "fullTimeEmployees"):
            if info.get(field) is None:
                P.log(f"    [companies] {t}: '{field}' not provided by Yahoo -> left blank")
        mc = info.get("marketCap")
        size = (pd.cut([float(mc)], cut_edges, labels=SIZE_LABELS)[0] if mc is not None else None)
        rows.append({
            "ticker": t, "company_name": name, "sector": sector, "sub_industry": sub,
            "exchange": info.get("exchange"), "cik": cik,
            "founded_year": P.first_year(founded) if founded else pd.NA,
            "size_category": size, "market_cap": mc,
            "revenue_ttm": info.get("totalRevenue"), "employees": info.get("fullTimeEmployees"),
            "selection_method": SELECTION_NEW,
        })
        P.log(f"    {t}: market_cap={mc:,} -> {size}" if mc is not None else f"    {t}: no market cap")
    comp_new = pd.DataFrame(rows)
    for c in ("cik", "founded_year", "market_cap", "revenue_ttm", "employees"):
        comp_new[c] = comp_new[c].astype("Int64")

    # --- Write: append-only; existing bytes preserved ------------------------------
    def eol(txt):  # keep each file's own line terminator
        return "\r\n" if "\r\n" in txt else "\n"

    nl = eol(comp_txt)
    lines = comp_txt.rstrip("\r\n").split(nl)
    comp_body = nl.join([lines[0] + ",selection_method"]
                        + [ln + "," + SELECTION_EXISTING for ln in lines[1:]]) + nl
    comp_body += comp_new.to_csv(index=False, header=False, lineterminator=nl)
    ev_body = ev_txt + ev_out.to_csv(index=False, header=False, lineterminator=eol(ev_txt))
    mr_body = mr_txt + mr_out.to_csv(index=False, header=False, float_format="%.6f",
                                     lineterminator=eol(mr_txt))

    # Validate everything in memory before anything touches disk.
    from io import StringIO
    chk_c = pd.read_csv(StringIO(comp_body), dtype=str, keep_default_na=False)
    chk_e = pd.read_csv(StringIO(ev_body), dtype=str, keep_default_na=False)
    chk_m = pd.read_csv(StringIO(mr_body), dtype=str, keep_default_na=False)
    assert list(chk_c.columns) == list(comp.columns) + ["selection_method"]
    assert len(chk_c) == len(comp) + len(comp_new)
    assert chk_c.iloc[: len(comp), :-1].equals(comp), "existing companies values changed"
    assert ev_body.startswith(ev_txt) and mr_body.startswith(mr_txt)
    assert len(chk_e) == len(chk_m) == len(ev_old) + len(ev_out)
    assert list(chk_e.columns) == ev_cols and list(chk_m.columns) == mr_cols
    assert chk_e["event_id"].is_unique and (chk_e["event_id"] == chk_m["event_id"]).all()
    assert chk_c["ticker"].is_unique

    comp_path.write_bytes(comp_body.encode("utf-8"))
    ev_path.write_bytes(ev_body.encode("utf-8"))
    mr_path.write_bytes(mr_body.encode("utf-8"))

    # --- Summary ---------------------------------------------------------------
    ev_all = pd.read_csv(ev_path)
    mr_all = pd.read_csv(mr_path)
    comp_all = pd.read_csv(comp_path)
    after = {"companies": len(comp_all), "events": len(ev_all), "market_reactions": len(mr_all)}
    cat_after = category_counts(ev_all["event_category"])

    P.log("\n[extension summary]")
    P.log(f"    {'file':<22}{'before':>8}{'after':>8}{'added':>8}")
    for k in before:
        P.log(f"    {k + '.csv':<22}{before[k]:>8}{after[k]:>8}{after[k] - before[k]:>+8}")
    P.log(f"    new event_ids: {ev_out['event_id'].iloc[0]} .. {ev_out['event_id'].iloc[-1]} "
          f"(EVT0001..EVT{last_id:04d} unchanged)")
    P.log("    new events per company: " + ", ".join(
        f"{t} {int((ev_out['ticker'] == t).sum())}" for t in NEW_TICKERS))
    P.log("    event_category breakdown, whole dataset (before -> after):")
    for c in sorted(set(cat_before) | set(cat_after), key=lambda c: -cat_after.get(c, 0)):
        b, a = cat_before.get(c, 0), cat_after.get(c, 0)
        P.log(f"      {c:<24}{b:>5} ->{a:>5}  ({a - b:+d})")
    P.log(f"    Security Incident events: {cat_after.get('Security Incident', 0)}")
    cyber = ev_out.merge(mr_out, on="event_id")
    cyber = cyber[cyber["event_category"] == "Security Incident"]
    for r in cyber.itertuples():
        P.log(f"      {r.event_id} {r.ticker:<5} report {r.report_date} t0={r.trading_date} "
              f"AR={r.abnormal_return:+.4f} CAR[-1,+1]={r.car_minus1_plus1:+.4f} "
              f"CAR[-1,+5]={r.car_minus1_plus5:+.4f} {r.reaction_category}")
    P.log("    selection_method: " + ", ".join(
        f"{k}={v}" for k, v in comp_all["selection_method"].value_counts().items()))
    P.log(f"    fundamentals for the 4 new companies fetched {date.today()} "
          f"(existing 57 were fetched 2026-09-16/29)")
    P.log(f"    runtime: {time.time() - t_start:.1f}s")

    with open(PROC / "run_log.txt", "a", encoding="utf-8", newline="\r\n") as f:
        f.write("\n" + "\n".join(P.LOG_LINES) + "\n")


if __name__ == "__main__":
    main()
