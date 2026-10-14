"""
EECS 4412 Phase 2 -- subset scatter plots: patterns hidden by the full dataset.

Writes charts 16-20 into ./output/.

  16  what happens after a big move, by company type (2x2 small multiples)
  17  Healthcare vs Financial Services, earnings (section 2) vs other events (section 8)
  18  Healthcare big pops: does a follow-up financing filing explain the fade?
  19  distress filings (section 3): delisting notices, share sales, rights changes
  20  earnings reaction vs volatility, one panel per sector

"Drift" = abnormal return over days +2..+5 = CAR[-1,+5] - CAR[-1,+1].
Binned medians always use every event whose x falls in the bin (y is never
clipped before computing them); only the drawing is windowed.

Run with: python subset_scatters.py   (from this folder, i.e. charts2/)
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from chart_style import (DATA, BLUE, ORANGE, AQUA, SURFACE, INK, INK2, MUTED, subtitle, save)

pct = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}%")

comp = pd.read_csv(DATA / "companies.csv", keep_default_na=False, na_values=[""])
ev = pd.read_csv(DATA / "events.csv", keep_default_na=False, na_values=[""], parse_dates=["report_date"])
mr = pd.read_csv(DATA / "market_reactions.csv")
df = ev.merge(mr, on="event_id").merge(
    comp[["ticker", "sector", "size_category", "volatility_tier", "trailing_volatility_1y"]], on="ticker")
df["car1"] = df["car_minus1_plus1"] * 100
df["car5"] = df["car_minus1_plus5"] * 100
df["drift"] = df["car5"] - df["car1"]
items = df["item_codes"].fillna("").str.split(",")
df["primary"] = items.apply(lambda l: ([i for i in l if i != "9.01"] or l or [""])[0])  # Phase 1 rule
df["section"] = df["primary"].str[0]
print(f"Phase 2 data: {len(df):,} events")


def drift_curve(sub, edges=np.arange(-40, 41, 2.5), min_n=30):
    """Median drift per CAR1 bin; returns bin centres, medians, counts."""
    g = sub.groupby(pd.cut(sub["car1"], edges), observed=True)["drift"].agg(["median", "size"])
    g = g[g["size"] >= min_n]
    return np.array([iv.mid for iv in g.index]), g["median"].to_numpy(), g["size"].to_numpy()


def plot_curve(ax, sub, color, label):
    x, y, n = drift_curve(sub)
    ax.plot(x, y, color=color, linewidth=2, label=f"{label} (n={len(sub):,})")
    ax.scatter(x, y, s=np.clip(n / n.max() * 120, 14, 120), color=color, edgecolors=SURFACE,
               linewidths=1.5, zorder=3)


def drift_axes(ax, title):
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title(title, fontsize=11.5)
    ax.legend(loc="lower left", frameon=False, fontsize=9)


# ============================================================
# CHART 16 -- after a big move: calm vs volatile company types
# ============================================================
pairs = [
    ("Volatility tier", df["volatility_tier"] == "Low", "Low volatility",
     df["volatility_tier"] == "Very High", "Very High volatility"),
    ("Company size", df["size_category"] == "Mega-cap", "Mega-cap", df["size_category"] == "Small", "Small"),
    ("Sector", df["sector"] == "Financial Services", "Financial Services",
     df["sector"] == "Healthcare", "Healthcare"),
    ("Earnings filings only (item 2.02)", (df["primary"] == "2.02") & (df["size_category"] == "Mega-cap"),
     "Mega-cap earnings", (df["primary"] == "2.02") & (df["size_category"] == "Small"), "Small-cap earnings"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, (title, calm, calm_lbl, wild, wild_lbl) in zip(axes.flat, pairs):
    plot_curve(ax, df[calm], BLUE, calm_lbl)
    plot_curve(ax, df[wild], ORANGE, wild_lbl)
    drift_axes(ax, title)
for ax in axes[1]:
    ax.set_xlabel("3-day reaction, CAR[-1,+1]")
for ax in axes[:, 0]:
    ax.set_ylabel("Median drift over days +2..+5")
fig.suptitle("After a big move, calm stocks bounce back; volatile stocks keep sinking",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.935, "Each dot = median of all events in a 2.5-pt reaction bin (bigger dot = more events). "
         "Above 0 = recovers, below 0 = keeps falling", ha="center", fontsize=10, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "16_subset_drift_by_company_type.png", tight=False)

# ============================================================
# CHART 17 -- Healthcare vs Financial Services, section 2 vs section 8
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for ax, (sec, name) in zip(axes, [("2", "Section 2: financial results (mostly earnings)"),
                                  ("8", "Section 8: other events (trial results, deals, news)")]):
    for sector, color in (("Financial Services", BLUE), ("Healthcare", ORANGE)):
        plot_curve(ax, df[(df["sector"] == sector) & (df["section"] == sec)], color, sector)
    drift_axes(ax, name)
    ax.set_xlabel("3-day reaction, CAR[-1,+1]")
axes[0].set_ylabel("Median drift over days +2..+5")
fig.suptitle("Finance settles after the filing; Healthcare keeps drifting down either way",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.905, "Your hypothesis, tested on 8-K item sections 2 and 8. Dots = bin medians; "
         "bigger dot = more events", ha="center", fontsize=10, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, "17_subset_healthcare_vs_finance_sections.png", tight=False)

# ============================================================
# CHART 18 -- Healthcare big pops and follow-up financing filings
# ============================================================
d = df.sort_values(["ticker", "report_date"]).copy()
gap = (d.groupby("ticker")["report_date"].shift(-1) - d["report_date"]).dt.days
nxt = d.groupby("ticker")["item_codes"].shift(-1).fillna("")
d["financing_next"] = gap.between(1, 7) & nxt.str.contains(r"1\.01|3\.02|8\.01")
hp = d[(d["sector"] == "Healthcare") & (d["car1"] > 10)]
fig, ax = plt.subplots(figsize=(10, 6.4))
for flag, color, lbl in ((False, BLUE, "No follow-up filing"),
                         (True, ORANGE, "Follow-up 8-K within 7 days (items 1.01, 3.02 or 8.01)")):
    s = hp[hp["financing_next"] == flag]
    ax.scatter(s["car1"], s["drift"], s=12, color=color, alpha=0.45,
               edgecolors=SURFACE, linewidths=0.3, label=f"{lbl}: n={len(s):,}, median drift {s['drift'].median():+.1f}%")
    x, y, _ = drift_curve(s, edges=np.arange(10, 101, 10), min_n=20)
    ax.plot(x, y, color=color, linewidth=2.5)
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xlim(10, 100)
ax.set_ylim(-60, 60)
off = int((~(hp["car1"].between(10, 100) & hp["drift"].between(-60, 60))).sum())
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("3-day reaction, CAR[-1,+1] (Healthcare events above +10%)")
ax.set_ylabel("Drift over days +2..+5")
ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=0.92, fontsize=9)
ax.set_title("Healthcare pops fade hardest when another filing follows", pad=18)
subtitle(ax, f"Lines = bin medians (all events). Follow-up items are typical of share offerings. "
             f"{off:,} points off-chart")
save(fig, "18_subset_healthcare_pop_then_filing.png", folder="healthcare")

# ============================================================
# CHART 19 -- distress filings (section 3)
# ============================================================
names = {"3.01": "3.01 Delisting / listing-rule notice", "3.02": "3.02 Unregistered sale of shares",
         "3.03": "3.03 Change to shareholder rights"}
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
W = 50
for ax, (code, name) in zip(axes, names.items()):
    s = df[df["primary"] == code]
    ax.scatter(s["car1"], s["car5"], s=10, color=BLUE, alpha=0.35,
               edgecolors=SURFACE, linewidths=0.3)
    ax.plot([-W, W], [-W, W], color=INK2, linewidth=1)
    g = s.groupby(pd.cut(s["car1"], np.arange(-W, W + 1, 5)), observed=True)["car5"].agg(["median", "size"])
    g = g[g["size"] >= 20]
    ax.plot([iv.mid for iv in g.index], g["median"], color=ORANGE, linewidth=2.5, label="Median 7-day reaction")
    below = (s["car5"] < s["car1"]).mean()
    ax.set_title(f"{name}\nn={len(s):,}  ·  median 7-day reaction {s['car5'].median():+.1f}%\n"
                 f"{below:.0%} fell further after day +1", fontsize=10.5)
    ax.set_xlim(-W, W)
    ax.set_ylim(-W, W)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("3-day reaction, CAR[-1,+1]")
axes[0].set_ylabel("7-day reaction, CAR[-1,+5]")
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("Distress filings: the diagonal is 'no change after day +1'; points below it kept falling",
             fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "19_subset_distress_filings.png", tight=False)

# ============================================================
# CHART 20 -- earnings reaction vs volatility, one panel per sector
# ============================================================
earn = df[df["primary"] == "2.02"]
co = earn.groupby("ticker").agg(med_abs=("car1", lambda s: s.abs().median()), n=("car1", "size"),
                                vol=("trailing_volatility_1y", "first"), sector=("sector", "first"))
co = co[(co["n"] >= 8) & co["sector"].notna() & (co["med_abs"] > 0)]
co["vol"] *= 100
lx, ly = np.log10(co["vol"]), np.log10(co["med_abs"])
slope, icept = np.polyfit(lx, ly, 1)
co["resid"] = ly - (icept + slope * lx)   # >0: reacts more than its volatility predicts
sectors = co.groupby("sector")["resid"].median().sort_values(ascending=False).index
xs = np.logspace(np.log10(10), np.log10(500), 50)
fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), sharex=True, sharey=True)
for ax, sec in zip(axes.flat, sectors):
    ax.scatter(co["vol"], co["med_abs"], s=6, color=MUTED, alpha=0.18, linewidths=0)
    s = co[co["sector"] == sec]
    ax.scatter(s["vol"], s["med_abs"], s=14, color=BLUE, alpha=0.7, edgecolors=SURFACE, linewidths=0.4)
    ax.plot(xs, 10 ** icept * xs ** slope, color=INK, linewidth=1.2)
    ratio = 10 ** s["resid"].median() - 1
    ax.set_title(f"{sec} ({len(s)})\n{ratio:+.0%} vs. what volatility predicts", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 500)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
for ax in axes.flat[len(sectors):]:
    ax.axis("off")
for ax in axes[-1]:
    ax.set_xlabel("Volatility (log)")
for ax in axes[:, 0]:
    ax.set_ylabel("Median |earnings reaction| (log)")
fig.suptitle("Earnings reactions by sector, after accounting for volatility",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.935, f"Each dot = a company's median |CAR[-1,+1]| on earnings filings (>= 8 of them). "
         f"Black line = all-company fit; panels sorted most to least reactive. "
         f"{int((~co['vol'].between(10, 500)).sum())} companies outside 10-500% volatility not drawn",
         ha="center", fontsize=10, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "20_subset_earnings_by_sector.png", tight=False)
