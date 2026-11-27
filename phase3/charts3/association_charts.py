"""
Charts for Phase 3 Task 1 (association). Run ../data_phase3/association_events.py first.

  01  the 20 most frequent itemsets (length >= 2) by support, the top 5 highlighted
  02  the 10 rules from the top 5 itemsets by confidence, the 2 highest and 2 lowest highlighted, with lift
  03  Supplementary Rule A: Late => each reaction inside Financial/Earnings, lift against the base rate and the profile-matched rate
  04  Supplementary Rule B: lift of size_category => reaction_category inside Financial/Earnings

Run:  ..\\..\\.venv\\Scripts\\python.exe association_charts.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_style import (BLUE, BLUE_RAMP, DATA, INK, INK2, LIFT, MUTED, ORANGE, REACTIONS, SIZE_ORDER, SURFACE, headline, heatmap_frame, lift_norm,
                         save)

SHORT = {"event_category": "event", "filing_timeliness": "timeliness", "size_category": "size", "volatility_tier": "volatility", "reaction_category": "reaction"}
RULE_A_CATEGORY = "Financial/Earnings"


def short(text):
    """{event_category=Financial/Earnings, filing_timeliness=On-time} -> event: Financial/Earnings + timeliness: On-time"""
    for long, brief in SHORT.items():
        text = text.replace(f"{long}=", f"{brief}: ")
    return text.strip("{}").replace(", ", " + ")


def top_itemsets():
    itemsets = pd.read_csv(DATA / "association_itemsets.csv").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    colours = [BLUE if rank <= 5 else BLUE_RAMP[1] for rank in itemsets["rank"]]
    ax.barh(range(len(itemsets)), itemsets["support"] * 100, color=colours, height=0.72, edgecolor=SURFACE, linewidth=2)
    for y, row in enumerate(itemsets.itertuples()):
        ax.text(row.support * 100 + 0.4, y, f"{row.support:.1%}", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(itemsets)), [f"{r.rank}. {short(r.itemset)}" for r in itemsets.itertuples()], fontsize=9.5)
    for label, rank in zip(ax.get_yticklabels(), itemsets["rank"]):
        label.set_color(INK if rank <= 5 else INK2)
        label.set_fontweight("bold" if rank <= 5 else "normal")
    ax.set_xlabel("Support (% of 411,440 filings)")
    ax.set_xlim(0, itemsets["support"].max() * 100 * 1.12)
    ax.grid(axis="y", visible=False)
    headline(fig, "The most frequent itemsets nearly all contain On-time, the item in 88% of filings",
             "Top 20 of the 1,120 itemsets of length 2 and 3 (items from different attributes). Dark bars: the top 5 used for the rules.")
    save(fig, "01_association_top_itemsets.png")


def rules_confidence():
    rules = pd.read_csv(DATA / "association_rules.csv", keep_default_na=False).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    colours = [BLUE if s == "highest confidence" else ORANGE if s == "lowest confidence" else BLUE_RAMP[0] for s in rules["selected_as"]]
    ax.barh(range(len(rules)), rules["confidence"] * 100, color=colours, height=0.7, edgecolor=SURFACE, linewidth=2)
    for y, row in enumerate(rules.itertuples()):
        ax.text(row.confidence * 100 + 1, y, f"{row.confidence:.1%}   lift {row.lift:.3f}", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(rules)), [short(r.antecedent) + "  =>  " + short(r.consequent) for r in rules.itertuples()], fontsize=9.5)
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, BLUE_RAMP[0])]
    ax.legend(handles, ["2 highest confidence", "2 lowest confidence", "Other rules"], frameon=False, loc="lower right", fontsize=9.5)
    headline(fig, "Confidence splits by direction, but every lift is close to 1",
             "The 10 rules from the top 5 itemsets. Rules into On-time inherit its 88% base rate; the reverse rules inherit the consequent's base rate.")
    save(fig, "02_association_rules_confidence.png")


def rule_a():
    rows = pd.read_csv(DATA / "association_rule_a_late.csv")
    rows = rows[rows["event_category"] == RULE_A_CATEGORY].set_index("reaction_category").loc[REACTIONS]
    x = np.arange(len(REACTIONS))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhline(1, color=MUTED, linewidth=1.2, zorder=1)
    ax.text(len(REACTIONS) - 0.55, 1.012, "lift 1 = no association", fontsize=9, color=INK2, ha="right", va="bottom")
    ax.scatter(x - 0.1, rows["lift"], s=90, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3, label="Lift vs the Financial/Earnings base rate")
    ax.scatter(x + 0.1, rows["lift_vs_profile_matched"], s=90, color=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=3, marker="D",
               label="Lift vs the rate matched on size x volatility")
    for xi, row in zip(x, rows.itertuples()):
        top = max(row.lift, row.lift_vs_profile_matched)
        ax.text(xi, top + 0.05, f"{row.lift:.2f} / {row.lift_vs_profile_matched:.2f}", ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(x, [f"{r}\n{rows.loc[r, 'rule_rows']:,} of {rows['antecedent_rows'].iloc[0]:,} Late" for r in REACTIONS], fontsize=9.5)
    ax.set_ylabel("Lift")
    ax.set_ylim(0.5, 1.75)
    ax.set_xlim(-0.6, len(REACTIONS) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    first = rows.iloc[0]
    headline(fig, "Late Financial/Earnings filings have fewer Strong reactions, also after matching", f"filing_timeliness=Late => reaction_category, inside {RULE_A_CATEGORY} ({first['rows_in_category']:,} filings, {first['antecedent_rows']:,} Late). "
                 f"Chi-square p = {first['chi2_p_value']:.1g}, Cramer's V {first['cramers_v']:.3f}.")
    save(fig, "03_association_rule_a_late.png")


def rule_b():
    rows = pd.read_csv(DATA / "association_rule_b_size.csv")
    lift = rows.pivot(index="size_category", columns="reaction_category", values="lift").reindex(index=SIZE_ORDER, columns=REACTIONS)
    counts = rows.pivot(index="size_category", columns="reaction_category", values="rule_rows").reindex(index=SIZE_ORDER, columns=REACTIONS)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    norm = lift_norm(lift.to_numpy())
    image = ax.imshow(lift.to_numpy(), cmap=LIFT, norm=norm, aspect="auto")
    for i, size in enumerate(SIZE_ORDER):
        for j, reaction in enumerate(REACTIONS):
            value = lift.loc[size, reaction]
            ax.text(j, i - 0.1, f"{value:.2f}", ha="center", va="center", fontsize=12, fontweight="bold", color=SURFACE if abs(value - 1) > 0.22 else INK)
            ax.text(j, i + 0.22, f"{counts.loc[size, reaction]:,} filings", ha="center", va="center", fontsize=8.5, color=SURFACE if abs(value - 1) > 0.22 else INK2)
    ax.set_xticks(range(len(REACTIONS)), REACTIONS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(SIZE_ORDER)), SIZE_ORDER)
    heatmap_frame(ax)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Lift (1 = the Financial/Earnings base rate)")
    bar.outline.set_visible(False)
    headline(fig, "Financial/Earnings: Small companies lean Strong Negative, Mega-caps moderate",
             f"size_category => reaction_category, {counts.to_numpy().sum():,} filings. Blue: more often than the Financial/Earnings base rate; red: less often.")
    save(fig, "04_association_rule_b_size.png")


if __name__ == "__main__":
    top_itemsets()
    rules_confidence()
    rule_a()
    rule_b()
