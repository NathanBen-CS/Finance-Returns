"""
EECS 4412 Phase 2 -- build processed/full_dataset.xlsx: one sequenced table per company.

  1. join Events (set 1) to Market_Reactions (set 3) on event_id
  2. join Companies (set 2) onto that by ticker
  3. sort by ticker, then report_date (ties: filing_date, then accession_number), and
     number each company's events 1..n in event_seq -- row 1 for a ticker is the first
     thing that happened to that company in the window

Values are written exactly as in the CSVs; returns are fractions shown with a % format.
Run with the project venv (needs openpyxl):  ..\\..\\.venv\\Scripts\\python.exe build_full_dataset.py
Writes processed/full_dataset.csv and processed/full_dataset.xlsx (same table).
Optional: --csv-only skips the xlsx; --limit N writes only the first N rows (smoke test, separate names).
"""
import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.indexed_list import IndexedList

PROC = Path(__file__).resolve().parent / "processed"
FONT = "Arial"

PCT_COLS = {"daily_return", "market_return", "abnormal_return", "car_minus1_plus1", "car_minus1_plus5",
            "volume_change_pct", "trailing_volatility_1y"}
INT_COLS = {"market_cap", "revenue_ttm", "employees"}
DATE_COLS = {"filing_date", "report_date", "trading_date"}


def load():
    read = dict(keep_default_na=False, na_values=[""])   # real tickers such as "NA" stay strings
    ev = pd.read_csv(PROC / "events.csv", **read)
    mr = pd.read_csv(PROC / "market_reactions.csv", **read)
    co = pd.read_csv(PROC / "companies.csv", **read)
    df = ev.merge(mr, on="event_id", how="inner", validate="one_to_one")          # step 1
    df = df.merge(co, on="ticker", how="left", validate="many_to_one")            # step 2
    assert len(df) == len(ev) == len(mr), "every event must have exactly one reaction"
    assert df["company_name"].notna().all(), "every event must match a company"
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c]).dt.date
    # whole-number columns with blanks load as float; keep them integers (no "3674.000000" in the CSV)
    for c in ("cik", "sic_code", "market_cap", "revenue_ttm", "employees", "num_items_disclosed"):
        df[c] = df[c].astype("Int64")
    df = df.sort_values(["ticker", "report_date", "filing_date", "accession_number"],
                        kind="mergesort").reset_index(drop=True)                     # step 3
    df.insert(0, "event_seq", df.groupby("ticker").cumcount() + 1)
    return ev, mr, co, df


def write_csv(df, path):
    """Same table as the xlsx; floats at the source CSVs' 6-decimal precision, dates ISO, blanks empty.
    Read it back with keep_default_na=False, na_values=[""] so tickers like "NA" stay text."""
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--csv-only", action="store_true", help="write full_dataset.csv only (skips the slow xlsx)")
    a = ap.parse_args()
    t0 = time.time()
    ev, mr, co, df = load()
    if a.limit:
        df = df.head(a.limit)
    write_csv(df, PROC / ("full_dataset.csv" if not a.limit else f"full_dataset_SAMPLE_{a.limit}.csv"))
    if a.csv_only:
        return
    cols = list(df.columns)
    out = PROC / ("full_dataset.xlsx" if not a.limit else f"full_dataset_SAMPLE_{a.limit}.xlsx")
    print(f"joined {len(df):,} rows x {len(cols)} columns -> {out.name}")

    wb = Workbook(write_only=True)
    wb._fonts = IndexedList([Font(name=FONT, size=10)])   # default font for every unstyled cell
    ws = wb.create_sheet("full_dataset")
    ws.freeze_panes = "C2"                                 # keep event_seq + event_id and the header visible
    widths = {c: max(len(c) + 2, 11) for c in cols}
    widths.update(company_name=30, sub_industry=28, sic_description=34, item_codes=14, accession_number=22,
                  event_category=22, reaction_category=16, selection_method=26, sector=22)
    for i, c in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths[c]
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(df) + 1}"

    head_font, head_fill = Font(name=FONT, size=10, bold=True, color="FFFFFF"), PatternFill("solid", fgColor="2A78D6")
    header = []
    for c in cols:
        cell = WriteOnlyCell(ws, value=c)
        cell.font, cell.fill = head_font, head_fill
        cell.alignment = Alignment(vertical="center")
        header.append(cell)
    ws.append(header)

    fmt = {c: ("0.00%" if c in PCT_COLS else "#,##0" if c in INT_COLS else "yyyy-mm-dd" if c in DATE_COLS
               else None) for c in cols}
    styled = [(i, fmt[c]) for i, c in enumerate(cols) if fmt[c]]
    for n, row in enumerate(df.itertuples(index=False, name=None), 1):
        vals = [None if (isinstance(v, float) and math.isnan(v)) else v for v in row]
        for i, f in styled:
            if vals[i] is not None:
                cell = WriteOnlyCell(ws, value=vals[i])
                cell.number_format = f
                vals[i] = cell
        ws.append(vals)
        if n % 50000 == 0:
            print(f"  {n:,} rows written ({time.time() - t0:.0f}s)", flush=True)

    notes = wb.create_sheet("notes")
    lines = [
        ("full_dataset.xlsx -- built", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Source files", "processed/events.csv, processed/market_reactions.csv, processed/companies.csv"),
        ("Step 1", "Events joined to Market_Reactions on event_id (one-to-one, checked)"),
        ("Step 2", "Companies joined by ticker (many-to-one, checked; every event matched a company)"),
        ("Step 3", "Sorted by ticker, then report_date; ties broken by filing_date, then accession_number"),
        ("event_seq", "1..n within each ticker: 1 = that company's first event in the window"),
        ("Rows", f"{len(df):,} (events.csv {len(ev):,}; market_reactions.csv {len(mr):,}; "
                 f"companies.csv {len(co):,} companies)"),
        ("Column order", "event_seq, then Events columns, then Market_Reactions, then Companies"),
        ("Number formats", "Returns and trailing_volatility_1y are fractions shown as % (0.0123 shows 1.23%); "
                           "values are unchanged from the CSVs"),
        ("Blank cells", "Values the sources do not have (e.g. volume_change_pct with zero baseline volume, "
                        "missing revenue/employees); nothing was filled in"),
        ("Built by", "phase2/data_phase2/build_full_dataset.py"),
    ]
    bold = Font(name=FONT, size=10, bold=True)
    notes.column_dimensions["A"].width = 26
    notes.column_dimensions["B"].width = 110
    for k, v in lines:
        kc = WriteOnlyCell(notes, value=k)
        kc.font = bold
        notes.append([kc, v])

    wb.save(out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
