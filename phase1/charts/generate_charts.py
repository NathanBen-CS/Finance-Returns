"""
EECS 4412 Phase 1 -- chart generation (matplotlib only).

Reads the 3 processed entity CSVs (../data/processed/companies.csv,
events.csv, market_reactions.csv) and produces the 5 required
visualizations -- 5 different dimensions, 5 different chart techniques --
into ./output/.

This script does ONLY charting. Task 6 statistics and the 220227286.xlsx
workbook are built by data/build_workbook.py, not here, so the two
concerns (visualize vs. produce the graded data deliverable) stay in
separate files.

Run with: python generate_charts.py   (from this folder, i.e. charts/)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA = Path("../data/processed")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ---- palette (from the dataviz skill's reference/palette.md) ----
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

comp = pd.read_csv(DATA / "companies.csv")
ev = pd.read_csv(DATA / "events.csv", parse_dates=["filing_date", "report_date"])
mr = pd.read_csv(DATA / "market_reactions.csv", parse_dates=["trading_date"])

df = ev.merge(mr, on="event_id", how="inner")
print("Merged events+reactions:", df.shape)

# ============================================================
# CHART 1 -- Bar chart: event_category frequency (nominal)
# ============================================================
counts = df["event_category"].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(counts.index, counts.values, color=BLUE, width=0.6)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 3, str(v), ha="center", color=INK, fontsize=10)
ax.set_ylabel("Number of events")
ax.set_title(f"Event category frequency (n={len(df)} 8-K filings)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "01_bar_event_category.png", dpi=160)
plt.close()

# ============================================================
# CHART 2 -- Donut chart: sector distribution (nominal) -- top 6 + Other
# ============================================================
sec_counts = comp["sector"].value_counts()
top6 = sec_counts.iloc[:6]
other = pd.Series({"Other sectors": sec_counts.iloc[6:].sum()})
donut_data = pd.concat([top6, other])
colors = [BLUE, ORANGE, AQUA, VIOLET, RED, GREEN, MUTED]
fig, ax = plt.subplots(figsize=(6.5, 6.5))
wedges, texts, autotexts = ax.pie(
    donut_data.values, labels=donut_data.index, autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
    colors=colors, startangle=90, wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=2),
    pctdistance=0.80, textprops={"color": INK, "fontsize": 9.5}
)
ax.set_title(f"Company sample by GICS sector (n={len(comp)})")
plt.tight_layout()
plt.savefig(OUT / "02_donut_sector.png", dpi=160)
plt.close()

# ============================================================
# CHART 3 -- Histogram: abnormal_return distribution (ratio)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["abnormal_return"] * 100, bins=24, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
ax.set_xlabel("Abnormal return on filing day (%)")
ax.set_ylabel("Number of events")
ax.set_title(f"Distribution of event-day abnormal returns (n={len(df)})")
plt.tight_layout()
plt.savefig(OUT / "03_hist_abnormal_return.png", dpi=160)
plt.close()

# ============================================================
# CHART 4 -- Boxplot: market_cap by size_category (ratio, grouped by ordinal)
# ============================================================
order = ["Small", "Mid", "Large", "Mega-cap"]
present = [c for c in order if c in comp["size_category"].dropna().unique()]
groups = [comp.loc[comp["size_category"] == c, "market_cap"].dropna() / 1e9 for c in present]
fig, ax = plt.subplots(figsize=(7, 4.5))
bp = ax.boxplot(groups, tick_labels=present, patch_artist=True, widths=0.5,
                 medianprops=dict(color=INK, linewidth=1.5),
                 boxprops=dict(facecolor=BLUE, alpha=0.55, edgecolor=INK2),
                 whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                 flierprops=dict(markerfacecolor=RED, markeredgecolor=RED, markersize=5, alpha=0.7))
ax.set_ylabel("Market capitalization (USD, billions)")
ax.set_yscale("log")
n_present = comp["market_cap"].notna().sum()
n_missing = comp["market_cap"].isna().sum()
title_suffix = f"n={n_present}" + (f", {n_missing} missing" if n_missing else ", complete")
ax.set_title(f"Market cap spread by size category ({title_suffix})")
plt.tight_layout()
plt.savefig(OUT / "04_box_marketcap.png", dpi=160)
plt.close()

# ============================================================
# CHART 5 -- Time series: events per week (interval: report_date)
# ============================================================
weekly = df.set_index("report_date").resample("W")["event_id"].count()
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(weekly.index, weekly.values, color=BLUE, linewidth=2)
ax.fill_between(weekly.index, weekly.values, color=BLUE, alpha=0.12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_ylabel("8-K filings that week")
ax.set_title(f"Filing volume over the event window ({df['report_date'].min().date()} to {df['report_date'].max().date()})")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "05_line_events_over_time.png", dpi=160)
plt.close()

print("5 charts written to", OUT.resolve())
