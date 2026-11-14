"""
Charts for Phase 2 Task 1 (PCA on the Companies entity). Run ../data_phase2/pca_companies.py first.

  23  scree plot: variance per component and the cumulative curve
  24  scores coloured by size_category: PC1 vs PC2 in full and zoomed, then PC1 vs PC3
  25  scores split by sector: PC1 vs PC2, one small panel per sector
  26  loadings of PC1-PC3 across the encoded columns

Run from inside charts2/:  ..\\..\\.venv\\Scripts\\python.exe pca_charts.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score

from chart_style import (BLUE, BLUE_RAMP, DATA, GRID, INK, INK2, MUTED, RED_RAMP, SIZE_ORDER, SURFACE,
                         category_axis, save, subtitle)

# Size tiers are ordered, so they share one hue from light to dark.
SIZE_COLOURS = dict(zip(SIZE_ORDER, [BLUE_RAMP[1], BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[6]]))
MIDPOINT = "#f0efec"   # the dataviz palette's neutral diverging midpoint
DIVERGING = LinearSegmentedColormap.from_list("red_blue", [RED_RAMP[5], RED_RAMP[3], MIDPOINT, BLUE_RAMP[3], BLUE_RAMP[5]])


def load():
    read = dict(keep_default_na=False, na_values=[""])
    scores = pd.read_csv(DATA / "pca_companies_scores.csv", **read)
    profile = pd.read_csv(DATA / "companies.csv", **read)[["ticker", "sector", "size_category"]]
    scores = scores.merge(profile, on="ticker", validate="one_to_one")
    variance = pd.read_csv(DATA / "pca_companies_explained_variance.csv")
    loadings = pd.read_csv(DATA / "pca_companies_loadings.csv", index_col="encoded_column")
    return scores, variance, loadings


def window(values):
    """Axis limits holding the middle 99% of the companies, with a small margin."""
    low, high = np.percentile(values, [0.5, 99.5])
    margin = (high - low) * 0.05
    return low - margin, high + margin


def scree(variance):
    share = variance["explained_variance_ratio"] * 100
    cumulative = variance["cumulative_explained_variance"] * 100
    x = np.arange(len(variance))
    kept = variance["kept"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x[kept], share[kept], width=0.72, color=BLUE, label="Kept component (top 3)")
    ax.bar(x[~kept], share[~kept], width=0.72, color=MUTED, label="Other component")
    ax.plot(x, cumulative, color=INK, linewidth=2, marker="o", markersize=4.5, label="Cumulative")
    ax.annotate(f"PC1-PC3: {cumulative[2]:.1f}%", (2, cumulative[2]), xytext=(4.0, cumulative[2] - 4),
                fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate("Two zero components:\neach one-hot group sums to 1", (len(x) - 1.5, 0.6),
                xytext=(len(x) - 7.5, 14), fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xticks(x, [c.replace("PC", "") for c in variance["component"]])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, 105)
    category_axis(ax)
    ax.legend(frameon=False, loc="center right")
    ax.set_title(f"The top 3 components explain {cumulative[2]:.1f}% of the variance", loc="left", pad=24)
    subtitle(ax, f"{len(variance)} encoded columns, all z-scored. Each bar is one component; the line is the running total.")
    save(fig, "23_pca_scree.png")


def scatter_by_size(scores, variance):
    share = variance["explained_variance_ratio"] * 100
    rng = np.random.default_rng(4412)
    shuffled = scores.iloc[rng.permutation(len(scores))]    # no tier is always drawn on top
    colours = shuffled["size_category"].map(SIZE_COLOURS)
    pc1_lim, pc2_lim, pc3_lim = window(scores["PC1"]), window(scores["PC2"]), window(scores["PC3"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.9))
    panels = [("PC2", None, None, "All companies"),
              ("PC2", pc1_lim, pc2_lim, "Zoomed to the middle 99%"),
              ("PC3", pc1_lim, pc3_lim, "PC1 vs PC3, zoomed")]
    for ax, (pc_y, xlim, ylim, name) in zip(axes, panels):
        ax.scatter(shuffled["PC1"], shuffled[pc_y], c=colours, s=15, linewidths=0.25, edgecolors=SURFACE)
        ax.set_xlabel(f"PC1 ({share[0]:.1f}% of variance)")
        ax.set_ylabel(f"{pc_y} ({share[int(pc_y[2]) - 1]:.1f}% of variance)")
        if xlim:
            ax.set(xlim=xlim, ylim=ylim)
            outside = (~scores["PC1"].between(*xlim) | ~scores[pc_y].between(*ylim)).sum()
            subtitle(ax, f"{outside} of {len(scores):,} companies fall outside this window")
        else:
            subtitle(ax, "Nothing hidden")
        ax.set_title(name, loc="left", pad=20)
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=7, color=SIZE_COLOURS[s], label=s) for s in SIZE_ORDER]
    fig.legend(handles=handles, title="size_category", frameon=False, ncol=4, loc="lower center")
    silhouette = silhouette_score(scores[["PC1", "PC2"]], scores["size_category"])
    fig.suptitle("Size tiers form a gradient along PC1, not separate clusters "
                 f"(silhouette {silhouette:+.2f})", x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    save(fig, "24_pca_scatter_size.png", tight=False)


def scatter_by_sector(scores, variance):
    share = variance["explained_variance_ratio"] * 100
    sectors = scores["sector"].value_counts().index
    xlim, ylim = window(scores["PC1"]), window(scores["PC2"])
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for ax, sector in zip(axes.flat, sectors):
        group = scores["sector"] == sector
        ax.scatter(scores.loc[~group, "PC1"], scores.loc[~group, "PC2"], color=GRID, s=8, linewidths=0)
        ax.scatter(scores.loc[group, "PC1"], scores.loc[group, "PC2"], color=BLUE, s=13, linewidths=0.25, edgecolors=SURFACE)
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_title(f"{sector} ({group.sum():,})", loc="left", fontsize=11)
    last = axes.flat[len(sectors)]
    last.axis("off")
    outside = (~scores["PC1"].between(*xlim) | ~scores["PC2"].between(*ylim)).sum()
    last.text(0, 0.95, f"Blue: the panel's sector.\nGrey: every other company.\n\n{outside} of {len(scores):,} companies fall\n"
              "outside this shared window.", va="top", fontsize=10.5, color=INK2)
    axes[1, -1].tick_params(labelbottom=True)    # the panel below it is the note, so this column labels its own axis
    fig.supxlabel(f"PC1 ({share[0]:.1f}% of variance)", fontsize=11)
    fig.supylabel(f"PC2 ({share[1]:.1f}% of variance)", fontsize=11)
    silhouette = silhouette_score(scores[["PC1", "PC2"]], scores["sector"])
    fig.suptitle(f"Sectors overlap heavily in the PC1-PC2 plane (silhouette {silhouette:+.2f})",
                 x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    save(fig, "25_pca_scatter_sector.png", tight=False)


def loadings_heatmap(loadings, variance):
    share = variance["explained_variance_ratio"] * 100
    fig, ax = plt.subplots(figsize=(8.2, 10))
    limit = np.abs(loadings.to_numpy()).max()
    image = ax.imshow(loadings.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    for row, values in enumerate(loadings.to_numpy()):
        for col, value in enumerate(values):
            ax.text(col, row, f"{round(value, 2) + 0.0:+.2f}", ha="center", va="center", fontsize=9.5,   # no "-0.00"
                    color=SURFACE if abs(value) > 0.65 * limit else INK)
    ax.set_xticks(range(3), [f"{pc}\n{share[i]:.1f}% of variance" for i, pc in enumerate(loadings.columns)])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(loadings)), loadings.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # rules between the numeric, sector and exchange groups
    groups = pd.Series(loadings.index).str.split("=").str[0].where(lambda s: s.isin(["sector", "exchange"]), "numeric")
    for row in np.flatnonzero(groups.to_numpy()[1:] != groups.to_numpy()[:-1]):
        ax.axhline(row + 0.5, color=SURFACE, linewidth=5)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Loading (red = negative, blue = positive)")
    bar.outline.set_visible(False)
    ax.set_title("What each component is made of", loc="left", pad=46)
    save(fig, "26_pca_loadings.png")


def main():
    scores, variance, loadings = load()
    scree(variance)
    scatter_by_size(scores, variance)
    scatter_by_sector(scores, variance)
    loadings_heatmap(loadings, variance)


if __name__ == "__main__":
    main()
