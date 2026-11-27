"""
Shared by the Phase 3 scripts: the filing table both tasks start from, and the workbook helpers.

The inputs are Phase 2's processed CSVs, read and never written. Every Phase 3 output goes to
data_phase3/processed/. The workbook helpers write in the same style as the Phase 1 and Phase 2
workbooks: a blue header row with white bold Arial, and Arial in the body.
"""
import itertools
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
PHASE2_PROC = HERE.parents[1] / "phase2" / "data_phase2" / "processed"
PROC = HERE / "processed"
PROC.mkdir(exist_ok=True)
WORKBOOK = PROC / "220227286.xlsx"

EXPECTED_FILINGS = 411_440
EXPECTED_COMPANIES = 3_900

# the attribute groups of the association table, each in its natural order
GROUPS = {
    "event_category": ["Financial/Earnings", "Management/Governance", "Other", "M&A/Contracts", "Legal/Regulatory", "Capital/Financing", "Security Incident"],
    "filing_timeliness": ["On-time", "Slightly Late", "Late"],
    "size_category": ["Small", "Mid", "Large", "Mega-cap"],
    "volatility_tier": ["Low", "Medium", "High", "Very High"],
    "reaction_category": ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"],
}
REACTIONS = GROUPS["reaction_category"]

SHEET_ORDER = ["T1_Association_Data", "T1_Association_Results", "T1_Association_Notes",
               "T2_Sequential_Data", "T2_Sequential_Results", "T2_Sequential_Notes"]

HEADER_FILL = PatternFill(start_color="2A78D6", end_color="2A78D6", fill_type="solid")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial")


def load_filings():
    """events + market_reactions (inner, on event_id), then the company columns (left, on ticker), without unreliable prices."""
    read = dict(keep_default_na=False, na_values=[""])   # real tickers such as "NA" stay strings
    events = pd.read_csv(PHASE2_PROC / "events.csv", parse_dates=["filing_date", "report_date"], **read)
    reactions = pd.read_csv(PHASE2_PROC / "market_reactions.csv", parse_dates=["trading_date"], **read)
    companies = pd.read_csv(PHASE2_PROC / "companies.csv", **read)
    df = events.merge(reactions, on="event_id", how="inner", validate="one_to_one")
    df = df.merge(companies[["ticker", "sector", "size_category", "volatility_tier"]], on="ticker", how="left", validate="many_to_one")
    assert df["price_data_unreliable"].dtype == bool
    merged, unreliable = len(df), int(df["price_data_unreliable"].sum())
    df = df[~df["price_data_unreliable"]].reset_index(drop=True)
    if len(df) != EXPECTED_FILINGS or df["ticker"].nunique() != EXPECTED_COMPANIES:
        raise SystemExit(f"Upstream data changed: expected {EXPECTED_FILINGS:,} filings from {EXPECTED_COMPANIES:,} companies, "
                         f"got {len(df):,} from {df['ticker'].nunique():,}. Stopping.")
    for col, values in GROUPS.items():
        assert df[col].isin(values).all(), f"{col} has blank or unexpected values"
    return df, merged, unreliable


def say_to(lines):
    """A print that also keeps the line, for the script's log file."""
    def say(text=""):
        print(text, flush=True)
        lines.append(text)
    return say


def percent(n, total):
    return f"{n:,} ({n / total:.2%})"


class Spinner:
    """Shows a spinner and the elapsed seconds on one console line while a slow step runs (console only, not the log)."""
    def __init__(self, text):
        self.text, self.done = text, threading.Event()

    def __enter__(self):
        self.start = time.time()
        def spin():
            for mark in itertools.cycle("|/-\\"):
                if self.done.wait(0.2):
                    break
                print(f"\r  {mark} {self.text} ... {time.time() - self.start:.0f}s ", end="", flush=True)
        self.thread = threading.Thread(target=spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()
        print(f"\r  {'failed' if exc[0] else 'done'}: {self.text} ({time.time() - self.start:.0f}s)          ", flush=True)


def progress(text, done, total):
    """One console line, rewritten in place, for a step that can count its own progress."""
    print(f"\r  {text}: {done:,} / {total:,} ({done / total:.0%}) ", end="" if done < total else "\n", flush=True)


def open_workbook():
    if WORKBOOK.exists():
        with Spinner(f"opening {WORKBOOK.name} (the other task's sheets are kept)"):
            wb = load_workbook(WORKBOOK)
    else:
        print(f"  {WORKBOOK.name} does not exist yet: starting a new workbook", flush=True)
        wb = Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    return wb


def save_workbook(wb):
    """Keep the T1 sheets before the T2 sheets whichever script ran last; any other sheet goes at the end."""
    wb._sheets.sort(key=lambda ws: SHEET_ORDER.index(ws.title) if ws.title in SHEET_ORDER else len(SHEET_ORDER))
    wb.active = 0
    with Spinner(f"saving {WORKBOOK.name} (the slowest step: a few minutes, with no way to show a percentage)"):
        wb.save(WORKBOOK)


def plain(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return None if value != value else value   # NaN stays a blank cell


def write_sheet(wb, name, df, formats=None, widths=None):
    """Replace one sheet with a single table."""
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    ws.append(list(df.columns))
    step = max(1, len(df) // 10)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        ws.append([plain(v) for v in row])
        if i % step == 0 or i == len(df):
            progress(f"{name}: rows written", i, len(df))
    for cell in ws[1]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    for col_idx, col in enumerate(df.columns, start=1):
        fmt = (formats or {}).get(col)
        for (cell,) in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
            cell.font = BODY_FONT
            if fmt:
                cell.number_format = fmt
        ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = (widths or {}).get(col, max(12, len(col) + 3))
        progress(f"{name}: columns formatted", col_idx, len(df.columns))
    ws.freeze_panes = "A2"
    return ws


def write_blocks(wb, name, blocks, widths):
    """One sheet of stacked tables, each under a bold title."""
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    row = 1
    for title, table, formats in blocks:
        ws.cell(row, 1, title).font = Font(name="Arial", bold=True)
        row += 1
        for col_idx, col in enumerate(table.columns, start=1):
            cell = ws.cell(row, col_idx, col)
            cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        for values in table.itertuples(index=False):
            row += 1
            for col_idx, (col, value) in enumerate(zip(table.columns, values), start=1):
                cell = ws.cell(row, col_idx, plain(value))
                cell.font = BODY_FONT
                if col in formats:
                    cell.number_format = formats[col]
        row += 2
    for letter, width in widths.items():
        ws.column_dimensions[letter].width = width
    return ws


def write_notes(wb, name, lines):
    ws = write_sheet(wb, name, pd.DataFrame({"notes": lines}), widths={"notes": 120})
    for (cell,) in ws.iter_rows(min_row=2):
        cell.alignment = Alignment(wrap_text=True, vertical="top")
