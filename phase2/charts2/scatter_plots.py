"""
EECS 4412 Phase 2 -- relationship scatter plots (matplotlib only).

Writes charts 10-15 into ./output/, alongside generate_charts.py's 01-09.

  10  CAR[-1,+1] vs CAR[-1,+5]                     (event level, density)
  11  CAR[-1,+1] vs post-event drift, days +2..+5  (event level, density)
  12  trailing volatility vs typical reaction size (company level)
  13  market cap vs trailing volatility            (company level)
  14  reaction size vs volume spike                (event level, density)
  15  trailing volatility vs late-filing share     (company level)

The relationships were chosen by measuring Spearman rank correlations first;
weak ones (filing lag, item count vs reaction size: |rho| < 0.08) are not plotted.
Spearman is reported because Pearson is dominated by the extreme-return outliers.
411,685 events cannot be drawn as individual dots, so event-level charts are
hexbin densities; company-level charts (3,900 points) are true scatters.
Display windows are stated on each chart; the data itself is never altered.

Run with: python scatter_plots.py   (from this folder, i.e. charts2/)
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

from chart_style import (DATA, PHASE1, BLUE, ORANGE, BLUE_RAMP, SURFACE, INK, INK2,
                         subtitle, usd, save)

DENSITY = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
pct = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}%")

comp = pd.read_csv(DATA / "companies.csv", keep_default_na=False, na_values=[""])
ev = pd.read_csv(DATA / "events.csv", keep_default_na=False, na_values=[""],
                 parse_dates=["filing_date", "report_date"])
mr = pd.read_csv(DATA / "market_reactions.csv")
df = ev.merge(mr, on="event_id")
df["car1"] = df["car_minus1_plus1"] * 100
df["car5"] = df["car_minus1_plus5"] * 100
df["drift"] = df["car5"] - df["car1"]                      # abnormal return over days +2..+5
p1_tickers = set(pd.read_csv(PHASE1 / "companies.csv")["ticker"])
print(f"Phase 2 data: {len(comp):,} companies, {len(df):,} events")


def rho(x, y):
    m = pd.concat([x, y], axis=1).dropna()
    return m.iloc[:, 0].corr(m.iloc[:, 1], method="spearman")


def binned_median(x, y, edges, min_n=50):
    """Median of y within each x bin (bins with fewer than min_n points are skipped)."""
    b = pd.cut(x, edges)
    g = pd.DataFrame({"b": b, "y": y}).groupby("b", observed=True)["y"].agg(["median", "size"])
    g = g[g["size"] >= min_n]
    centers = np.array([iv.mid for iv in g.index])
    return centers, g["median"].to_numpy()


def density(ax, x, y, extent, **kw):
    hb = ax.hexbin(x, y, gridsize=110, extent=extent, cmap=DENSITY, norm=LogNorm(), mincnt=1,
                   linewidths=0, **kw)
    cb = ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Events per hexagon (log scale)")
    cb.outline.set_visible(False)
    return hb


def windowed(x, y, lo, hi):
    keep = x.between(lo, hi) & y.between(lo, hi)
    return x[keep], y[keep], int((~keep).sum())


# ============================================================
# CHART 10 -- CAR[-1,+1] vs CAR[-1,+5]
# ============================================================
W = 30
x, y, out = windowed(df["car1"], df["car5"], -W, W)
fig, ax = plt.subplots(figsize=(8, 6.4))
density(ax, x, y, (-W, W, -W, W))
ax.plot([-W, W], [-W, W], color=INK2, linewidth=1)
inx = df["car1"].between(-W, W)  # medians use every event in the x-window; clipping y would bias edge bins
cx, my = binned_median(df.loc[inx, "car1"], df.loc[inx, "car5"], np.arange(-W, W + 1, 1.5))
ax.plot(cx, my, color=ORANGE, linewidth=2)
ax.set_xlim(-W, W)
ax.set_ylim(-W, W)
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("CAR[-1,+1]: 3-day reaction")
ax.set_ylabel("CAR[-1,+5]: 7-day reaction")
ax.legend(handles=[Line2D([], [], color=ORANGE, lw=2, label="Median CAR[-1,+5] per CAR[-1,+1] bin"),
                   Line2D([], [], color=INK2, lw=1, label="y = x (no change after day +1)")],
          loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("The 7-day reaction tracks the 3-day reaction one-for-one", pad=18)
subtitle(ax, f"rho = {rho(df['car1'], df['car5']):+.2f}, partly built in: days -1..+1 count in both. "
             f"Outside ±{W}%: {out:,} events")
save(fig, "10_scatter_car1_vs_car5.png")

# ============================================================
# CHART 11 -- CAR[-1,+1] vs drift over days +2..+5
# ============================================================
x, y, out = windowed(df["car1"], df["drift"], -W, W)
fig, ax = plt.subplots(figsize=(8, 6.4))
density(ax, x, y, (-W, W, -W, W))
ax.axhline(0, color=INK2, linewidth=1)
inx = df["car1"].between(-W, W)
cx, my = binned_median(df.loc[inx, "car1"], df.loc[inx, "drift"], np.arange(-W, W + 1, 1.5))
ax.plot(cx, my, color=ORANGE, linewidth=2)
ax.set_xlim(-W, W)
ax.set_ylim(-W, W)
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("CAR[-1,+1]: initial 3-day reaction")
ax.set_ylabel("Abnormal return over days +2 to +5 (CAR5 - CAR1)")
ax.legend(handles=[Line2D([], [], color=ORANGE, lw=2, label="Median later drift per reaction bin")],
          loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("No follow-through: the reaction is complete by day +1", pad=18)
subtitle(ax, f"rho = {rho(df['car1'], df['drift']):+.3f}: big moves neither continue nor reverse. "
             f"Outside ±{W}%: {out:,} events")
save(fig, "11_scatter_car1_vs_drift.png")

# ============================================================
# company-level frame
# ============================================================
co = comp.set_index("ticker").join(df.assign(abs_car1=df["car1"].abs()).groupby("ticker").agg(
    mean_abs_car1=("abs_car1", "mean"),
    late_share=("filing_timeliness", lambda s: (s != "On-time").mean() * 100)))
co["vol"] = co["trailing_volatility_1y"] * 100
co["p1"] = co.index.isin(p1_tickers)


def company_scatter(ax, x, y, label_all):
    ax.scatter(x[~co["p1"]], y[~co["p1"]], s=14, color=BLUE, alpha=0.35, edgecolors=SURFACE,
               linewidths=0.4, label=label_all)
    ax.scatter(x[co["p1"]], y[co["p1"]], s=44, color=ORANGE, edgecolors=SURFACE, linewidths=1,
               label=f"Phase 1 companies ({int(co['p1'].sum())})", zorder=3)


def loglog_fit(ax, x, y):
    lx, ly = np.log10(x), np.log10(y)
    ok = np.isfinite(lx) & np.isfinite(ly)
    slope, icept = np.polyfit(lx[ok], ly[ok], 1)
    xs = np.logspace(lx[ok].min(), lx[ok].max(), 50)
    ax.plot(xs, 10 ** icept * xs ** slope, color=INK, linewidth=1.5, label=f"Log-log fit (slope {slope:.2f})")
    return slope


# ============================================================
# CHART 12 -- trailing volatility vs typical reaction size
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 6))
company_scatter(ax, co["vol"], co["mean_abs_car1"], f"Phase 2 companies ({len(co):,})")
slope = loglog_fit(ax, co["vol"], co["mean_abs_car1"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("Trailing 1-year realized volatility (annualized, log scale)")
ax.set_ylabel("Average |CAR[-1,+1]| across the company's events (log scale)")
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("Volatile companies react more to every filing", pad=18)
subtitle(ax, f"rho = {rho(co['vol'], co['mean_abs_car1']):+.2f}, the strongest link in the data. "
             f"Slope {slope:.2f}: 10% more volatility ~ {slope * 10:.0f}% bigger moves")
save(fig, "12_scatter_volatility_vs_reaction.png")

# ============================================================
# CHART 13 -- market cap vs trailing volatility
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 6))
company_scatter(ax, co["market_cap"], co["vol"], f"Phase 2 companies ({len(co):,})")
slope = loglog_fit(ax, co["market_cap"], co["vol"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(usd))
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("Market capitalization (USD, log scale)")
ax.set_ylabel("Trailing 1-year realized volatility (log scale)")
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax.set_title("Bigger companies are calmer", pad=18)
subtitle(ax, f"Spearman rho = {rho(co['market_cap'], co['vol']):+.2f}. Phase 1's S&P 500 picks cluster in the "
             f"large, low-volatility corner")
save(fig, "13_scatter_marketcap_vs_volatility.png")

# ============================================================
# CHART 14 -- reaction size vs volume spike
# ============================================================
vr = 1 + df["volume_change_pct"]                 # event-day volume / 10-day baseline
aar = df["abnormal_return"].abs() * 100
keep = (vr > 0) & (aar > 0) & vr.notna()
vx, vy = vr[keep], aar[keep]
fig, ax = plt.subplots(figsize=(8, 6.4))
hb = ax.hexbin(vx, vy, xscale="log", yscale="log", gridsize=90, extent=(-1.5, 2.5, -3, 2.3),
               cmap=DENSITY, norm=LogNorm(), mincnt=1, linewidths=0)
cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Events per hexagon (log scale)")
cb.outline.set_visible(False)
cx, my = binned_median(vx, vy, np.logspace(-1.5, 2.5, 33))
ax.plot(cx, my, color=ORANGE, linewidth=2)
ax.axvline(1, color=INK2, linewidth=1)
ax.set_xlim(10 ** -1.5, 10 ** 2.5)
ax.set_ylim(10 ** -3, 10 ** 2.3)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}x"))
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("Event-day volume vs prior 10-day average (log scale; 1x = normal)")
ax.set_ylabel("|Abnormal return| on event day (log scale)")
ax.legend(handles=[Line2D([], [], color=ORANGE, lw=2, label="Median |abnormal return| per volume bin")],
          loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("Bigger price moves come with bigger volume spikes", pad=18)
small_low = (df.loc[keep & (vr < 0.3), "ticker"].map(comp.set_index("ticker")["size_category"]) == "Small").mean()
subtitle(ax, f"rho = {rho(vx, vy):+.2f}. Bump below 0.3x = illiquid small caps ({small_low:.0%} of events there)")
fig.text(0.01, 0.01, f"n={len(vx):,}; excludes {int((~keep).sum()):,} events with zero event-day volume or "
         f"blank volume change", fontsize=8.5, color=INK2)
save(fig, "14_scatter_reaction_vs_volume.png")

# ============================================================
# CHART 15 -- trailing volatility vs late-filing share
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.scatter(co["vol"], co["late_share"], s=14, color=BLUE, alpha=0.35, edgecolors=SURFACE,
           linewidths=0.4, label=f"Companies ({len(co):,})")
edges = np.quantile(co["vol"], np.linspace(0, 1, 11))
dec = co.groupby(pd.cut(co["vol"], edges, include_lowest=True), observed=True).agg(
    v=("vol", "median"), late=("late_share", "mean"))
ax.plot(dec["v"], dec["late"], color=ORANGE, linewidth=2, marker="o", markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, label="Average per volatility decile")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.set_ylim(-2, 102)
ax.set_xlabel("Trailing 1-year realized volatility (annualized, log scale)")
ax.set_ylabel("Share of the company's 8-Ks not On-time")
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("More volatile companies file late more often", pad=18)
subtitle(ax, f"Spearman rho = {rho(co['vol'], co['late_share']):+.2f}. Decile averages rise from "
             f"{dec['late'].iloc[0]:.0f}% to {dec['late'].iloc[-1]:.0f}% (Phase 1 calendar-day buckets)")
save(fig, "15_scatter_volatility_vs_late_filing.png")

print("6 scatter charts written")
