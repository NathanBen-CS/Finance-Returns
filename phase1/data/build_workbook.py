"""
EECS 4412 Phase 1 -- Task 6 statistics + submission workbook builder.

Reads the 3 processed entity CSVs (./processed/companies.csv, events.csv,
market_reactions.csv) and:

  1. runs the Task 6 basic statistical analysis (1 nominal, 1 ordinal,
     1 interval, 1 ratio) -> writes ./processed/task6_stats.json
  2. builds the required submission workbook, 220227286.xlsx, with 7
     worksheets: the 3 entity tables (Companies, Events, Market_Reactions)
     plus the 4 Task 6 dimension tables, each on its own worksheet
     (T1_Nominal_EventCategory, T1_Ordinal_ReactionCategory,
     T1_Interval_ReportDate, T1_Ratio_AbnormalReturn). The course outline
     requires the raw data to live in this workbook rather than in the
     report document or a standalone CSV.

This is intentionally separate from generate_charts.py (../charts/) --
that script only does matplotlib visualization; this one only does the
graded data deliverable. Both read the same processed CSVs so neither can
drift out of sync with the other.

Run with: python build_workbook.py   (from this folder, i.e. data/)
The output is written as 220227286.xlsx and is ready to submit as-is.
"""
import pandas as pd
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

PROC = Path("processed")

comp = pd.read_csv(PROC / "companies.csv")
ev = pd.read_csv(PROC / "events.csv", parse_dates=["filing_date", "report_date"])
mr = pd.read_csv(PROC / "market_reactions.csv", parse_dates=["trading_date"])
df = ev.merge(mr, on="event_id", how="inner")
print("Merged events+reactions:", df.shape)

# ============================================================
# TASK 6 -- Basic statistical analysis
# Nominal: event_category | Ordinal: reaction_category
# Interval: report_date   | Ratio: abnormal_return
# ============================================================
results = {}

# --- Nominal: event_category ---
mode_cat = df["event_category"].mode()[0]
freq = df["event_category"].value_counts()
rel_freq = (freq / freq.sum() * 100).round(1)
results["nominal_event_category"] = {
    "mode": mode_cat,
    "mode_count": int(freq[mode_cat]),
    "frequency_table": freq.to_dict(),
    "relative_frequency_pct": rel_freq.to_dict(),
    "num_categories": df["event_category"].nunique(),
}

# --- Ordinal: reaction_category ---
order_map = {"Strong Negative": 1, "Negative": 2, "Neutral": 3, "Positive": 4, "Strong Positive": 5}
df["_reaction_rank"] = df["reaction_category"].map(order_map)
mode_reac = df["reaction_category"].mode()[0]
median_rank = df["_reaction_rank"].median()
inv_map = {v: k for k, v in order_map.items()}
median_label = inv_map.get(median_rank, f"between ranks (median rank={median_rank})")
results["ordinal_reaction_category"] = {
    "mode": mode_reac,
    "median_rank": float(median_rank),
    "median_label": median_label,
    "frequency_table": df["reaction_category"].value_counts().to_dict(),
}

# --- Interval: report_date ---
rd = df["report_date"]
results["interval_report_date"] = {
    "min": str(rd.min().date()), "max": str(rd.max().date()),
    "range_days": int((rd.max() - rd.min()).days),
    "median": str(rd.median().date()),
}

# --- Ratio: abnormal_return ---
ar = df["abnormal_return"] * 100  # in percent
results["ratio_abnormal_return"] = {
    "mean_pct": round(float(ar.mean()), 3),
    "std_pct": round(float(ar.std()), 3),
    "min_pct": round(float(ar.min()), 3),
    "max_pct": round(float(ar.max()), 3),
    "n": int(ar.shape[0]),
}

print(json.dumps(results, indent=2, default=str))
with open(PROC / "task6_stats.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# ============================================================
# WORKBOOK -- 220227286.xlsx
# ============================================================
HEADER_FILL = PatternFill(start_color="2A78D6", end_color="2A78D6", fill_type="solid")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial")


def write_sheet(wb, name, frame):
    ws = wb.create_sheet(name)
    for row in dataframe_to_rows(frame, index=False, header=True):
        ws.append(row)
    for cell in ws[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = BODY_FONT
    ws.freeze_panes = "A2"
    # autosize columns (blank/NaN cells render as "" so they don't widen the column)
    for i, col in enumerate(frame.columns, start=1):
        cell_lens = frame[col].map(lambda v: len(str(v)) if pd.notna(v) else 0)
        width = max(len(str(col)), cell_lens.max() if len(frame) else 0)
        ws.column_dimensions[get_column_letter(i)].width = min(width + 3, 45)
    return ws


t1_nominal = df[["event_id", "event_category"]].rename(
    columns={"event_category": "event_category_nominal"})
t1_ordinal = df[["event_id", "reaction_category"]].rename(
    columns={"reaction_category": "reaction_category_ordinal"})
t1_interval = df[["event_id", "report_date"]].copy()
t1_interval["report_date"] = t1_interval["report_date"].dt.date
t1_interval = t1_interval.rename(columns={"report_date": "report_date_interval"})
t1_ratio = df[["event_id", "abnormal_return"]].rename(
    columns={"abnormal_return": "abnormal_return_ratio"})

wb = Workbook()
wb.remove(wb.active)  # drop the default empty sheet
write_sheet(wb, "Companies", comp)
write_sheet(wb, "Events", ev)
write_sheet(wb, "Market_Reactions", mr)
write_sheet(wb, "T1_Nominal_EventCategory", t1_nominal)
write_sheet(wb, "T1_Ordinal_ReactionCategory", t1_ordinal)
write_sheet(wb, "T1_Interval_ReportDate", t1_interval)
write_sheet(wb, "T1_Ratio_AbnormalReturn", t1_ratio)

xlsx_path = Path("220227286.xlsx")
wb.save(xlsx_path)
print(f"Wrote {xlsx_path.resolve()} with 7 worksheets "
      f"(Companies, Events, Market_Reactions, 4x T1_* dimension sheets)")
