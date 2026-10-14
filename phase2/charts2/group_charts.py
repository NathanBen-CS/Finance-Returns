"""
EECS 4412 Phase 2 -- the same chart set for every company group, for side-by-side comparison.

output/shared/       all companies (plus cross-sector extras 21-22)
output/<sector>/     one folder per sector (healthcare, financial_services, ...)

Every folder gets the same five charts, with the same file names and the SAME axis
limits for each chart type, so shapes can be compared by flipping between folders:

  A_drift_after_big_moves      after a big 3-day move, does the stock recover or keep going?
  B_event_type_map             each filing type: typical direction (x) vs typical size (y)
  C_followup_reaction_link     does the next filing's reaction echo this one's?
  D_drift_when_followed        drift when another filing lands inside days +2..+5, vs none
  E_what_follows_what          which filing type tends to come next (within 30 days)

Colour = company style: blue = calm (Low/Medium volatility tier), red = volatile
(High/Very High). "Drift" = abnormal return over days +2..+5 (CAR[-1,+5] - CAR[-1,+1]).
A follow-up is the company's next 8-K on a strictly later date (same-day filings share
one price reaction). Medians use every event in an x-bin; nothing is clipped first.

Run with: python group_charts.py   (from this folder, i.e. charts2/)
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from chart_style import (CALM, VOLATILE, CALM_LIGHT, VOLATILE_LIGHT, BLUE_RAMP, RED_RAMP,
                         SURFACE, INK, INK2, MUTED, subtitle, save)
from chart_data import load_events, binned_median, slug, CATEGORY_ORDER, SHORT

pct = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}%")
# Narrow bins where events are dense, wide bins in the tails so each tail point averages enough events
# to show a shape rather than one odd stock.
DRIFT_EDGES = np.array([-40, -30, -20, -15, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 20, 30, 40])
LINK_EDGES = np.arange(-30, 31, 3)
IN_WINDOW = (2, 5)      # follow-up's event day falls on day +2..+5 of this event (drift window)
LINK_GAP = (3, 20)      # >= 3 trading days apart so the two 3-day windows never overlap
STYLES = [("calm", CALM, CALM_LIGHT, "Calm (Low/Medium volatility)"),
          ("volatile", VOLATILE, VOLATILE_LIGHT, "Volatile (High/Very High)")]

comp, df = load_events()
sectors = df["sector"].dropna().value_counts().index
GROUPS = [("shared", "All companies", df)] + [(slug(s), s, df[df["sector"] == s]) for s in sectors]
print(f"{len(df):,} events, {len(GROUPS) - 1} sectors")


def counts_note(sub):
    c = sub.groupby("style")["ticker"].nunique()
    return (f"{len(sub):,} events; {c.get('calm', 0):,} calm and {c.get('volatile', 0):,} volatile companies")


def ylims(values, pad=0.4, lo_cap=-9, hi_cap=7):
    v = np.concatenate([np.asarray(x) for x in values if len(x)])
    return max(v.min() - pad, lo_cap), min(v.max() + pad, hi_cap)


def bubbles(ax, x, y, n, color, nmax, label=None, marker="o", lw=2):
    ax.plot(x, y, color=color, linewidth=lw, label=label)
    ax.scatter(x, y, s=np.clip(n / nmax * 110, 12, 110), color=color, marker=marker,
               edgecolors=SURFACE, linewidths=1.5, zorder=3)


def zero_axes(ax):
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)


# ---------------------------------------------------------------- precompute (global axis limits)
A = {f: {st: binned_median(s["car1"], s["drift"], DRIFT_EDGES) for st, s in sub.groupby("style")}
     for f, _, sub in GROUPS}
A_LIM = ylims([c[1] for g in A.values() for c in g.values()])

A2 = {f: {cat: {st: binned_median(s["car1"], s["drift"], DRIFT_EDGES)
                for st, s in sub[sub["event_category"] == cat].groupby("style")}
           for cat in CATEGORY_ORDER} for f, _, sub in GROUPS}
A2_LIM = ylims([c[1] for g in A2.values() for cats in g.values() for c in cats.values()])

B = {}
for f, _, sub in GROUPS:
    g = sub.groupby(["style", "event_category"]).agg(x=("car1", "median"), y=("car1", lambda s: s.abs().median()),
                                                     n=("car1", "size"))
    B[f] = g[g["n"] >= 30]
allB = pd.concat(B.values())
B_X = (allB["x"].min() - 0.4, allB["x"].max() + 0.4)
B_Y = (0, allB["y"].max() * 1.12)

link = df["next_gap_tdays"].between(*LINK_GAP)
C = {f: {st: binned_median(s["car1"], s["next_car1"], LINK_EDGES)
         for st, s in sub[link.loc[sub.index]].groupby("style")} for f, _, sub in GROUPS}
C_LIM = ylims([c[1] for g in C.values() for c in g.values()])

followed = df["next_gap_tdays"].between(*IN_WINDOW)
D = {}
for f, _, sub in GROUPS:
    D[f] = {}
    for st, s in sub.groupby("style"):
        fl = followed.loc[s.index]
        D[f][st] = (binned_median(s.loc[fl, "car1"], s.loc[fl, "drift"], DRIFT_EDGES, min_n=20),
                    binned_median(s.loc[~fl, "car1"], s.loc[~fl, "drift"], DRIFT_EDGES),
                    int(fl.sum()), int((~fl).sum()))
D_LIM = ylims([c[0][1] for g in D.values() for c in g.values()] + [c[1][1] for g in D.values() for c in g.values()])


# ---------------------------------------------------------------- A
def chart_A(folder, label, sub):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    nmax = max(c[2].max() for c in A[folder].values() if len(c[2]))
    for st, color, _, name in STYLES:
        if st in A[folder]:
            x, y, n = A[folder][st]
            bubbles(ax, x, y, n, color, nmax, label=f"{name}: {int((sub['style'] == st).sum()):,} events")
    zero_axes(ax)
    ax.set_xlim(-41, 41)
    ax.set_ylim(*A_LIM)
    ax.set_xlabel("3-day reaction to the filing, CAR[-1,+1]")
    ax.set_ylabel("Median drift over days +2..+5")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.set_title(f"{label}: after a big move, recover or keep going?", pad=18)
    subtitle(ax, "Dot = median of all events in a reaction bin (wider bins in the tails; bigger = more events). Above 0 = recovers. "
                 "Same axes in every folder")
    save(fig, "A_drift_after_big_moves.png", folder)


# ---------------------------------------------------------------- A2: chart A, one panel per filing type
def chart_A2(folder, label, sub):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6), sharex=True, sharey=True)
    for ax, cat in zip(axes.flat, CATEGORY_ORDER):
        curves = {st: c for st, c in A2[folder][cat].items() if len(c[0])}
        n_cat = int((sub["event_category"] == cat).sum())
        if curves:
            nmax = max(c[2].max() for c in curves.values())
            for st, color, _, _ in STYLES:
                if st in curves:
                    bubbles(ax, *curves[st], color, nmax, lw=1.8)
        else:
            ax.text(0, 0.5, "too few events", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=10, color=INK2)
        zero_axes(ax)
        ax.set_xlim(-41, 41)
        ax.set_ylim(*A2_LIM)
        ax.set_title(f"{SHORT[cat][0]} · {SHORT[cat]} ({n_cat:,})", fontsize=11)
    key = axes.flat[len(CATEGORY_ORDER)]
    key.axis("off")
    key.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", label=nm) for _, c, _, nm in STYLES],
               loc="center", frameon=False, fontsize=10.5, title="Company style", title_fontsize=10.5)
    for ax in axes[-1]:
        ax.set_xlabel("3-day reaction, CAR[-1,+1]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median drift +2..+5")
    fig.suptitle(f"{label}: after a big move, by type of filing", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925, "Chart A split by filing type. Above 0 = recovers, below 0 = keeps going. Bins need "
             ">= 30 events; same axes in every folder", ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "A2_drift_by_filing_type.png", folder, tight=False)


# ---------------------------------------------------------------- B
def chart_B(folder, label, sub):
    g = B[folder]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for cat in CATEGORY_ORDER:
        pts = {st: g.loc[(st, cat)] for st, *_ in STYLES if (st, cat) in g.index}
        if len(pts) == 2:
            ax.plot([pts["calm"]["x"], pts["volatile"]["x"]], [pts["calm"]["y"], pts["volatile"]["y"]],
                    color=MUTED, linewidth=1, zorder=1)
        for st, color, _, _ in STYLES:
            if st in pts:
                p = pts[st]
                # letter inside the dot labels every point without colliding text
                ax.scatter(p["x"], p["y"], s=330, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
                ax.text(p["x"], p["y"], SHORT[cat][0], ha="center", va="center", fontsize=9,
                        fontweight="bold", color="#ffffff", zorder=4)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(*B_X)
    ax.set_ylim(*B_Y)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Typical direction: median CAR[-1,+1]   (left = bad news, right = good news)")
    ax.set_ylabel("Typical size: median |CAR[-1,+1]|")
    ax.legend(handles=[Line2D([], [], marker="o", color=c, linestyle="", markersize=9, label=nm)
                       for _, c, _, nm in STYLES], loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title(f"{label}: how each kind of filing moves the stock", pad=18)
    subtitle(ax, "E Earnings · G Governance · M M&A · O Other · L Legal · F Financing · C Cyber. "
                 "Line joins a type's two dots")
    save(fig, "B_event_type_map.png", folder)


# ---------------------------------------------------------------- C
def chart_C(folder, label, sub):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    curves = C[folder]
    nmax = max(c[2].max() for c in curves.values() if len(c[2]))
    sl = sub[link.loc[sub.index]]
    for st, color, _, name in STYLES:
        if st in curves and len(curves[st][0]):
            s = sl[sl["style"] == st]
            rho = s["car1"].corr(s["next_car1"], method="spearman")
            x, y, n = curves[st]
            bubbles(ax, x, y, n, color, nmax, label=f"{name}: {len(s):,} pairs, rho {rho:+.2f}")
    zero_axes(ax)
    ax.set_xlim(-31, 31)
    ax.set_ylim(*C_LIM)
    ax.set_xlabel("Reaction to this filing, CAR[-1,+1]")
    ax.set_ylabel("Median reaction to the company's next filing")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title(f"{label}: does the next filing's reaction echo this one?", pad=18)
    subtitle(ax, f"Next 8-K {LINK_GAP[0]}-{LINK_GAP[1]} trading days later (windows never overlap). "
                 "Rising line = reactions cluster")
    save(fig, "C_followup_reaction_link.png", folder)


# ---------------------------------------------------------------- D
def chart_D(folder, label, sub):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, (st, color, light, name) in zip(axes, STYLES):
        if st not in D[folder]:
            ax.axis("off")
            continue
        (fx, fy, fn), (nx, ny, nn), nf, nnf = D[folder][st]
        nmax = max([v.max() for v in (fn, nn) if len(v)] or [1])
        if len(nx):
            bubbles(ax, nx, ny, nn, light, nmax, label=f"No filing in days +2..+5 ({nnf:,})", marker="s")
        if len(fx):
            bubbles(ax, fx, fy, fn, color, nmax, label=f"Another filing in days +2..+5 ({nf:,})")
        zero_axes(ax)
        ax.set_xlim(-41, 41)
        ax.set_ylim(*D_LIM)
        ax.set_title(name, fontsize=11.5)
        ax.set_xlabel("3-day reaction to the filing, CAR[-1,+1]")
        ax.legend(loc="lower left", frameon=False, fontsize=9)
    axes[0].set_ylabel("Median drift over days +2..+5")
    fig.suptitle(f"{label}: does a follow-up filing change what happens next?", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.905, "Dark = the company filed again inside the drift window; light = it did not. "
             "Same axes in every folder", ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "D_drift_when_followed.png", folder, tight=False)


# ---------------------------------------------------------------- E
def chart_E(folder, label, sub):
    cols = CATEGORY_ORDER + ["None in 30 days"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, (st, color, _, name), ramp in zip(axes, STYLES, (BLUE_RAMP, RED_RAMP)):
        s = sub[sub["style"] == st]
        nxt = np.where(s["next_gap_days"] <= 30, s["next_category"], "None in 30 days")
        t = pd.crosstab(s["event_category"], nxt).reindex(index=CATEGORY_ORDER, columns=cols, fill_value=0)
        rows = t.sum(axis=1)
        share = t.div(rows.where(rows > 0), axis=0) * 100
        share[rows < 30] = np.nan                                    # too few to read a pattern
        cmap = LinearSegmentedColormap.from_list(st, ramp)
        cmap.set_bad("#f0efec")
        ax.imshow(np.ma.masked_invalid(share.values), cmap=cmap, vmin=0, vmax=60, aspect="auto")
        ax.grid(False)
        for i in range(len(CATEGORY_ORDER)):
            for j in range(len(cols)):
                v = share.values[i, j]
                if np.isnan(v):
                    continue
                r, g_, b, _ = cmap(min(v, 60) / 60)
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color=INK if 0.2126 * r + 0.7152 * g_ + 0.0722 * b > 0.5 else "#ffffff")
        ax.set_xticks(range(len(cols)), [SHORT.get(c, "None\n<=30d") for c in cols], fontsize=9)
        ax.set_yticks(range(len(CATEGORY_ORDER)),
                      [f"{SHORT[c]} ({int(rows[c]):,})" for c in CATEGORY_ORDER], fontsize=9.5)
        ax.set_xlabel("Next filing within 30 days")
        ax.set_title(name, fontsize=11.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("This filing (events)")
    fig.suptitle(f"{label}: which filing tends to come next?", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.91, "Each row sums to 100%: share of filings of that type followed by each type within "
             "30 days. Grey = fewer than 30 events. Colour capped at 60%", ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "E_what_follows_what.png", folder, tight=False)


for folder, label, sub in GROUPS:
    chart_A(folder, label, sub)
    chart_A2(folder, label, sub)
    chart_B(folder, label, sub)
    chart_C(folder, label, sub)
    chart_D(folder, label, sub)
    chart_E(folder, label, sub)

# ---------------------------------------------------------------- shared extra 21: all sectors at once
fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
for ax, sec in zip(axes.flat, sectors):
    curves = A[slug(sec)]
    nmax = max(c[2].max() for c in curves.values() if len(c[2]))
    for st, color, _, _ in STYLES:
        if st in curves:
            bubbles(ax, *curves[st], color, nmax, lw=1.6)
    zero_axes(ax)
    ax.set_xlim(-41, 41)
    ax.set_ylim(*A_LIM)
    ax.set_title(f"{sec} ({int((df['sector'] == sec).sum()):,})", fontsize=10.5)
leg = axes.flat[len(sectors)]
leg.axis("off")
leg.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", label=nm) for _, c, _, nm in STYLES],
           loc="center", frameon=False, fontsize=11, title="Company style", title_fontsize=11)
for ax in axes[-1]:
    ax.set_xlabel("3-day reaction")
for ax in axes[:, 0]:
    ax.set_ylabel("Median drift +2..+5")
fig.suptitle("Every sector side by side: after a big move, recover or keep going?", fontsize=15, fontweight="bold")
fig.text(0.5, 0.94, "Blue = calm companies, red = volatile. Above 0 = recovers after the move; below 0 = keeps "
         "falling. Same axes as each sector's A chart", ha="center", fontsize=10.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "21_sector_grid_drift.png", "shared", tight=False)

# ---------------------------------------------------------------- shared extra 22: drift by follow-up type
rows = []
for st, *_ in STYLES:
    s = df[df["style"] == st]
    fl = followed.loc[s.index]
    rows.append((st, "No filing in days +2..+5", s.loc[~fl, "drift"].median(), int((~fl).sum())))
    for cat, g in s[fl].groupby("next_category"):
        rows.append((st, cat, g["drift"].median(), len(g)))
dd = pd.DataFrame(rows, columns=["style", "next", "drift", "n"])
dd = dd[dd["n"] >= 30]
order = ["No filing in days +2..+5"] + [c for c in CATEGORY_ORDER if c in set(dd["next"])]
fig, ax = plt.subplots(figsize=(10, 5.6))
for i, cat in enumerate(order):
    pts = dd[dd["next"] == cat].set_index("style")
    if len(pts) == 2:
        ax.plot(pts["drift"], [i, i], color=MUTED, linewidth=1.5, zorder=1)
    for st, color, _, _ in STYLES:
        if st in pts.index:
            ax.scatter(pts.loc[st, "drift"], i, s=90, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.annotate(f"{pts.loc[st, 'n']:,}", (pts.loc[st, "drift"], i), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, linewidth=1)
ax.set_yticks(range(len(order)), [o if o.startswith("No") else f"Followed by {SHORT[o]}" for o in order])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(pct)
ax.set_xlabel("Median drift over days +2..+5 (numbers = events)")
ax.legend(handles=[Line2D([], [], marker="o", color=c, linestyle="", markersize=9, label=nm)
                   for _, c, _, nm in STYLES], loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("What the follow-up filing is matters: drift by type of next filing", pad=18)
subtitle(ax, "All companies. Next 8-K's event day falls on day +2..+5, inside this filing's drift window")
save(fig, "22_drift_by_followup_type.png", "shared")
print("done")
