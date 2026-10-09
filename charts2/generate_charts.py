"""
EECS 4412 Phase 2 -- chart generation (matplotlib only).

Reads the Phase 2 entity CSVs (../data_phase2/processed/companies.csv,
events.csv, market_reactions.csv) and writes PNGs into ./output/.

  01-05  the five Phase 1 charts, recreated on the Phase 2 data
  06-09  new charts for what Phase 2 added: volatility tiers, filing
         timeliness across company size, and Phase 1 vs Phase 2 calmness

Display-only adaptations for the larger, heavier-tailed data are stated on
each chart (histogram window, monthly bins); the data itself is never altered.

Run with: python generate_charts.py   (from this folder, i.e. charts2/)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

DATA = Path("../data_phase2/processed")
PHASE1 = Path("../data/processed")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ---- palette (dataviz skill reference palette; categorical slots in fixed order) ----
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
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
    "axes.axisbelow": True,  # gridlines behind marks
})

SIZE_ORDER = ["Small", "Mid", "Large", "Mega-cap"]
VOL_ORDER = ["Low", "Medium", "High", "Very High"]


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def category_axis(ax):
    ax.grid(axis="x", visible=False)  # vertical rules add nothing on a category axis


def usd(x, _=None):
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            v = x / div
            return f"${v:,.0f}{unit}" if v >= 10 or v == int(v) else f"${v:.1f}{unit}"
    return f"${x:,.0f}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("  wrote", name)


# keep_default_na=False: tickers such as "NA" are real symbols, not missing values
comp = pd.read_csv(DATA / "companies.csv", keep_default_na=False, na_values=[""])
ev = pd.read_csv(DATA / "events.csv", keep_default_na=False, na_values=[""],
                 parse_dates=["filing_date", "report_date"])
mr = pd.read_csv(DATA / "market_reactions.csv", parse_dates=["trading_date"])
df = ev.merge(mr, on="event_id", how="inner").merge(
    comp[["ticker", "size_category", "volatility_tier"]], on="ticker", how="left")
print(f"Phase 2 data: {len(comp):,} companies, {len(df):,} events")

# ============================================================
# CHART 1 -- Bar: event_category frequency (nominal)
# ============================================================
counts = df["event_category"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(counts.index, counts.values, color=BLUE, width=0.6)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + counts.max() * 0.012, f"{v:,}",
            ha="center", color=INK, fontsize=9.5)
category_axis(ax)
ax.set_ylabel("Number of events")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_title(f"Event category frequency (n={len(df):,} 8-K filings)", pad=18)
subtitle(ax, "Security Incident (Item 1.05) exists only since Dec 2023, hence its small count")
plt.xticks(rotation=20, ha="right")
save(fig, "01_bar_event_category.png")

# ============================================================
# CHART 2 -- Donut: sector distribution (nominal), top 6 + Other
# ============================================================
sec = comp["sector"].fillna("(no sector)").value_counts()
donut = pd.concat([sec.iloc[:6], pd.Series({"Other sectors": sec.iloc[6:].sum()})])
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(donut.values, labels=[f"{k}\n{v:,}" for k, v in donut.items()],
       autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
       colors=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, MUTED], startangle=90, counterclock=False,
       wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=2),
       pctdistance=0.79, textprops={"color": INK, "fontsize": 9.5})
ax.set_title(f"Companies by sector (Yahoo classification, n={len(comp):,})")
other = ", ".join(sec.index[6:])
fig.text(0.5, 0.04, f"Other sectors: {other}", ha="center", fontsize=8.5, color=INK2, wrap=True)
save(fig, "02_donut_sector.png")

# ============================================================
# CHART 3 -- Histogram: abnormal_return distribution (ratio)
# ============================================================
ar = df["abnormal_return"] * 100
LIM = 25
inside = ar[ar.between(-LIM, LIM)]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.hist(inside, bins=np.arange(-LIM, LIM + 0.5, 0.5), color=BLUE, edgecolor=SURFACE, linewidth=0.4)
ax.axvline(0, color=INK2, linewidth=1)
ax.set_xlabel("Abnormal return on event day (%)")
ax.set_ylabel("Number of events")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_title(f"Distribution of event-day abnormal returns (n={len(ar):,})", pad=18)
subtitle(ax, f"Shown: -{LIM}% to +{LIM}%, 0.5-pt bins. {len(ar) - len(inside):,} events "
             f"({(len(ar) - len(inside)) / len(ar):.1%}) lie outside; median {ar.median():+.2f}%, "
             f"std {ar.std():.1f} pts")
save(fig, "03_hist_abnormal_return.png")

# ============================================================
# CHART 4 -- Boxplot: market_cap by size_category (ratio by ordinal)
# ============================================================
groups = [comp.loc[comp["size_category"] == c, "market_cap"] for c in SIZE_ORDER]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.boxplot(groups, tick_labels=SIZE_ORDER, patch_artist=True, widths=0.5,
           medianprops=dict(color=INK, linewidth=1.5),
           boxprops=dict(facecolor=BLUE, alpha=0.55, edgecolor=INK2),
           whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
           flierprops=dict(marker="o", markerfacecolor=MUTED, markeredgecolor=MUTED,
                           markersize=3, alpha=0.5))
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(usd))
category_axis(ax)
ax.set_ylabel("Market capitalization (USD, log scale)")
ax.set_title(f"Market cap spread by size category (n={len(comp):,}, quartiles)", pad=18)
mc = comp["market_cap"]
subtitle(ax, f"Range spans {usd(mc.min())} to {usd(mc.max())}: the Phase 2 universe reaches far "
             "below the S&P 500".replace("$", r"\$"))
save(fig, "04_box_marketcap.png")

# ============================================================
# CHART 5 -- Time series: events per month (interval)
# ============================================================
monthly = df.set_index("report_date").resample("MS")["event_id"].count()
last = df["report_date"].max()
if last != last + pd.offsets.MonthEnd(0):  # final month is partial -> drop it, no fake cliff
    monthly = monthly.iloc[:-1]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(monthly.index, monthly.values, color=BLUE, linewidth=2)
ax.fill_between(monthly.index, monthly.values, color=BLUE, alpha=0.12)
ax.set_xlim(monthly.index.min(), monthly.index.max())
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("8-K filings that month")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_title(f"Filing volume over the event window "
             f"({monthly.index.min():%b %Y} to {monthly.index.max():%b %Y})", pad=18)
subtitle(ax, "Complete months only. May peaks = Q1 earnings + annual-meeting votes (Item 5.07)")
save(fig, "05_line_events_over_time.png")

# ============================================================
# CHART 6 -- Heatmap: size_category x volatility_tier (companies)
# ============================================================
xt = pd.crosstab(comp["size_category"], comp["volatility_tier"]).reindex(
    index=SIZE_ORDER, columns=VOL_ORDER, fill_value=0)
cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(xt.values, cmap=cmap, aspect="auto", vmin=0)
ax.grid(False)
ax.set_xticks(range(len(VOL_ORDER)), VOL_ORDER)
ax.set_yticks(range(len(SIZE_ORDER)), SIZE_ORDER)
ax.set_xlabel("Volatility tier (trailing 1-year realized volatility)")
ax.set_ylabel("Size category (market cap)")
for i in range(len(SIZE_ORDER)):
    for j in range(len(VOL_ORDER)):
        v = xt.values[i, j]
        r, g, b, _ = cmap(v / xt.values.max())
        light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11, color=INK if light else "#ffffff")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Companies")
cb.outline.set_visible(False)
ax.set_title(f"Smaller companies are far more volatile (n={len(comp):,} companies)", pad=18)
subtitle(ax, "Each tier holds 25% of companies; size and volatility move in opposite directions")
save(fig, "06_heatmap_size_x_volatility.png")

# ============================================================
# CHART 7 -- Bar: share of events not On-time, by size_category
# ============================================================
late = df.groupby("size_category", observed=True)["filing_timeliness"].apply(
    lambda s: (s != "On-time").mean() * 100).reindex(SIZE_ORDER)
sl = df[df["filing_timeliness"] == "Slightly Late"]
bd = np.busday_count(sl["report_date"].values.astype("datetime64[D]"),
                     sl["filing_date"].values.astype("datetime64[D]"))
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(late.index, late.values, color=BLUE, width=0.6)
for b, v in zip(bars, late.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}%", ha="center", color=INK, fontsize=10)
category_axis(ax)
ax.set_ylabel("Events filed > 4 calendar days after event (%)")
ax.set_xlabel("Size category")
overall = (df["filing_timeliness"] != "On-time").mean()
ax.set_title(f"Smaller companies file later (all events: {overall:.1%} not On-time)", pad=18)
subtitle(ax, f"Phase 1 calendar-day buckets. {(bd <= 4).mean():.0%} of 'Slightly Late' filings were "
             f"within SEC's 4-business-day deadline")
save(fig, "07_bar_late_filing_by_size.png")

# ============================================================
# CHART 8 -- Histogram: trailing volatility, Phase 2 pool vs Phase 1 companies
# ============================================================
p1 = pd.read_csv(PHASE1 / "companies.csv")
vol = comp["trailing_volatility_1y"] * 100
p1_vol = comp.loc[comp["ticker"].isin(p1["ticker"]), "trailing_volatility_1y"] * 100
edges = vol.quantile([0.25, 0.5, 0.75]).values
LO, HI = 5, 1000  # display window; shares stay relative to each full group
bins = np.logspace(np.log10(LO), np.log10(HI), 46)
fig, ax = plt.subplots(figsize=(10, 5))
# each group as % of itself, so 61 and 3,900 companies are comparable on ONE axis
ax.hist(vol.clip(LO * 1.001, HI * 0.999), bins=bins, weights=np.full(len(vol), 100 / len(vol)),
        color=BLUE, edgecolor=SURFACE, linewidth=0.4, label=f"Phase 2 universe ({len(vol):,} companies)")
ax.hist(p1_vol, bins=bins, weights=np.full(len(p1_vol), 100 / len(p1_vol)), histtype="step",
        color=ORANGE, linewidth=2, label=f"Phase 1 companies ({len(p1_vol)}, same measure)")
ax.set_xscale("log")
ax.set_xlim(LO, HI)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}%"))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ymax = ax.get_ylim()[1] * 1.12
ax.set_ylim(0, ymax)
for e in edges:
    ax.axvline(e, color=INK2, linewidth=1)
bounds = [LO, *edges, HI]
for k, name in enumerate(VOL_ORDER):  # tier names centred in their band, staggered so narrow bands don't collide
    x = np.sqrt(bounds[k] * bounds[k + 1])
    ax.text(x, ymax * (0.98 if k % 2 == 0 else 0.91), name, ha="center", va="top", fontsize=9.5, color=INK2)
ax.set_xlabel("Trailing 1-year realized volatility (annualized, log scale)")
ax.set_ylabel("Share of group's companies")
ax.legend(loc="center right", frameon=False)
counts_p1 = comp.loc[comp["ticker"].isin(p1["ticker"]), "volatility_tier"].value_counts().reindex(VOL_ORDER)
ax.set_title("Phase 1's S&P 500 sample sits in the calm end of the market", pad=18)
subtitle(ax, "Phase 1 by Phase 2 tier: " + ", ".join(f"{k} {v}" for k, v in counts_p1.items())
         + f". End bins also hold {int((vol < LO).sum())} companies below {LO}% and "
           f"{int((vol > HI).sum())} above {HI:,}%")
save(fig, "08_hist_volatility_phase1_vs_phase2.png")

# ============================================================
# CHART 9 -- Boxplot: CAR[-1,+1] by volatility_tier
# ============================================================
car = [df.loc[df["volatility_tier"] == t, "car_minus1_plus1"] * 100 for t in VOL_ORDER]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.boxplot(car, tick_labels=VOL_ORDER, patch_artist=True, widths=0.5, showfliers=False,
           medianprops=dict(color=INK, linewidth=1.5),
           boxprops=dict(facecolor=BLUE, alpha=0.55, edgecolor=INK2),
           whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
ax.axhline(0, color=INK2, linewidth=1)
ax.set_ylabel("CAR[-1,+1] (%)")
ax.set_xlabel("Volatility tier")
iqr = [np.subtract(*np.percentile(c, [75, 25])) for c in car]
for i, (c, w) in enumerate(zip(car, iqr), 1):
    ax.text(i + 0.3, np.percentile(c, 75), f"IQR {w:.1f} pts", va="bottom", fontsize=9.5, color=INK2)
category_axis(ax)
ax.set_title("Event reactions widen sharply with volatility tier", pad=18)
subtitle(ax, f"Box = middle 50%, whiskers = 1.5 x IQR, outliers not drawn (n={len(df):,} events)")
save(fig, "09_box_car_by_volatility.png")

print(f"9 charts written to {OUT.resolve()}")
