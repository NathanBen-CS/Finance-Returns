"""
Charts for Phase 3 Task 2 (sequential). Run ../data_phase3/sequential_events.py first.

  05  lift of every previous -> next reaction pattern (all consecutive filings), with confidence and counts
  06  lift of the 2 most frequent patterns in every variant (main, robustness A and B, no shared CAR day)
  07  share of transitions with the same reaction as the previous filing, against a within-company shuffle and independence

Run:  ..\\..\\.venv\\Scripts\\python.exe sequential_charts.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_style import BLUE, BLUE_RAMP, DATA, INK, INK2, LIFT, MUTED, ORANGE, REACTIONS, SURFACE, headline, heatmap_frame, lift_norm, save

MAIN = "All consecutive filings"
LABELS = {MAIN: "All consecutive filings", "Robustness A: same event_category only": "A: same event_category only",
          "Robustness B: short gap": "B: short gap (<= median)", "Robustness B: long gap": "B: long gap (> median)"}


def label(variant):
    return LABELS.get(variant, variant.replace("Extra: ", "Extra: ").replace(" trading days", " days"))


def transition_lift():
    patterns = pd.read_csv(DATA / "sequential_patterns.csv")
    main = patterns[patterns["variant"] == MAIN]
    lift = main.pivot(index="prev_reaction", columns="curr_reaction", values="lift").reindex(index=REACTIONS, columns=REACTIONS)
    confidence = main.pivot(index="prev_reaction", columns="curr_reaction", values="confidence").reindex(index=REACTIONS, columns=REACTIONS)
    counts = main.pivot(index="prev_reaction", columns="curr_reaction", values="count").reindex(index=REACTIONS, columns=REACTIONS)
    top = main.nsmallest(2, "rank")
    fig, ax = plt.subplots(figsize=(10.5, 7))
    image = ax.imshow(lift.to_numpy(), cmap=LIFT, norm=lift_norm(lift.to_numpy()), aspect="auto")
    for i, a in enumerate(REACTIONS):
        for j, b in enumerate(REACTIONS):
            value = lift.loc[a, b]
            dark = abs(value - 1) > 0.4
            ax.text(j, i - 0.17, f"{value:.2f}", ha="center", va="center", fontsize=12, fontweight="bold", color=SURFACE if dark else INK)
            ax.text(j, i + 0.13, f"conf {confidence.loc[a, b]:.1%}", ha="center", va="center", fontsize=8.5, color=SURFACE if dark else INK2)
            ax.text(j, i + 0.33, f"{counts.loc[a, b]:,}", ha="center", va="center", fontsize=8.5, color=SURFACE if dark else INK2)
    for row in top.itertuples():
        i, j = REACTIONS.index(row.prev_reaction), REACTIONS.index(row.curr_reaction)
        ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=INK, linewidth=2))
    ax.set_xticks(range(len(REACTIONS)), REACTIONS)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Next filing's reaction", labelpad=8)
    ax.set_yticks(range(len(REACTIONS)), REACTIONS)
    ax.set_ylabel("Previous filing's reaction")
    heatmap_frame(ax)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Lift (confidence / base rate of the next reaction)")
    bar.outline.set_visible(False)
    total = main["count"].sum()
    headline(fig, "Every reaction repeats on the company's next filing more often than its base rate",
             f"{total:,} consecutive pairs of a company's filings. Outlined: the 2 most frequent patterns. Part of the diagonal comes from pairs that share CAR days (charts 06 and 07).")
    save(fig, "05_sequential_transition_lift.png")


def robustness_lift():
    patterns = pd.read_csv(DATA / "sequential_patterns.csv")
    summary = pd.read_csv(DATA / "sequential_summary.csv")
    variants = list(summary["variant"])
    tracked = list(patterns[patterns["variant"] == MAIN].nsmallest(2, "rank")["pattern"])
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axvline(1, color=MUTED, linewidth=1.2, zorder=1)
    for pattern, colour, marker, shift in zip(tracked, (BLUE, ORANGE), ("o", "D"), (-0.12, 0.12)):
        part = patterns[patterns["pattern"] == pattern].set_index("variant").loc[variants]
        y = np.arange(len(variants)) + shift
        ax.scatter(part["lift"], y, s=85, color=colour, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=pattern.replace("<", "").replace(">", "").replace(" - ", " -> "))
        for xi, yi in zip(part["lift"], y):
            ax.text(xi + 0.012, yi, f"{xi:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(variants)), [f"{label(v)}\n{n:,} transitions" for v, n in zip(variants, summary["transitions"])], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0.95, 1.75)
    ax.set_xlabel("Lift of the rule (1 = no more often than the base rate)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    headline(fig, "Persistence is strongest for close filings and fades with distance",
             "The 2 most frequent patterns of the main table, recomputed in each variant. Pairs with gap 0 share their CAR, so they always repeat.")
    save(fig, "06_sequential_robustness.png")


def same_reaction():
    summary = pd.read_csv(DATA / "sequential_summary.csv")
    shuffled = summary["same_reaction_within_company_shuffle"].dropna().iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(summary))
    ax.barh(y, summary["same_reaction_share"] * 100, color=BLUE, height=0.62, edgecolor=SURFACE, linewidth=2)
    for yi, value in zip(y, summary["same_reaction_share"]):
        ax.text(value * 100 + 0.3, yi, f"{value:.1%}", va="center", fontsize=9.5, color=INK2)
    independent = summary["same_reaction_if_independent"].iloc[0]
    ax.axvline(independent * 100, color=MUTED, linewidth=1.5, linestyle="--")
    ax.axvline(shuffled * 100, color=ORANGE, linewidth=1.8)
    ax.text(independent * 100 - 0.3, len(summary) - 0.35, f"independent\n{independent:.1%}", ha="right", va="center", fontsize=9, color=INK2)
    ax.text(shuffled * 100 + 0.3, len(summary) - 0.35, f"within-company\nshuffle {shuffled:.1%}", ha="left", va="center", fontsize=9, color=ORANGE)
    ax.set_yticks(y, [label(v) for v in summary["variant"]], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.set_ylim(len(summary) - 0.05, -0.6)
    ax.set_xlabel("Transitions where the next filing has the same reaction_category (%)")
    ax.grid(axis="y", visible=False)
    headline(fig, "Without shared CAR days, repeats sit close to what each company's mix alone gives", "Shuffling keeps each company's reactions but not their order (20 shuffles). Independent: the next reaction ignores the company and the previous one.")
    save(fig, "07_sequential_same_reaction.png")


if __name__ == "__main__":
    transition_lift()
    robustness_lift()
    same_reaction()
