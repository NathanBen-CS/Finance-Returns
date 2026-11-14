"""
Charts for Phase 2 Task 3 (k-means). Run ../data_phase2/clustering_events.py first.

  29  elbow: SSE per k, k=2 to 10, with the required k=2, 3, 4, 5 marked
  30  silhouette per k for the same range
  31  the k=5 clusters: share of each cluster by size_category, exchange and reaction_category
  32  Task 1's PC1 vs PC2, one point per company, coloured by k=5 cluster (compare with chart 24)
  33  market_cap vs CAR[-1,+1] on a fixed sample of events, coloured by k=5 cluster

Charts 32 and 33 add their joins and counts to the end of processed/clustering_events_log.txt.

Run from inside charts2/:  ..\\..\\.venv\\Scripts\\python.exe clustering_charts.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score

from chart_style import AQUA, BLUE, BLUE_RAMP, DATA, INK, INK2, MAGENTA, MUTED, ORANGE, SIZE_ORDER, SURFACE, YELLOW, save, subtitle, usd
from pca_charts import window

CLASSES = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]
EXCHANGES = ["NYQ", "NMS", "NGM", "NCM", "ASE", "PNK"]
REQUIRED = [2, 3, 4, 5]
SHARES = LinearSegmentedColormap.from_list("share", [SURFACE, *BLUE_RAMP])   # one hue: darker means a larger share
CLUSTER_SLOTS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]   # fixed categorical order, assigned from the largest mean market_cap down
SAMPLE = 20_000
SAMPLE_SEED = 42        # the convention of the silhouette sample
CAR_LIMIT = 50          # percent; points beyond it are off the chart, not out of the sample
LOG_MARKER = "SUPPLEMENTARY CHARTS 32 AND 33"


def by_k(ax, metrics, column):
    """The curve for k=2 to 10, with the required k drawn as filled points."""
    required = metrics[metrics["k"].isin(REQUIRED)]
    context = metrics[~metrics["k"].isin(REQUIRED)]
    ax.plot(metrics["k"], metrics[column], color=MUTED, linewidth=2, zorder=1)
    ax.scatter(context["k"], context[column], s=60, color=SURFACE, edgecolors=MUTED, linewidths=1.6, zorder=2, label="Context only (k=6 to 10)")
    ax.scatter(required["k"], required[column], s=80, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3, label="Required (k=2, 3, 4, 5)")
    ax.set_xticks(metrics["k"])
    ax.set_xlabel("Number of clusters, k")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    return required


def elbow(metrics):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    required = by_k(ax, metrics, "sse")
    for row in required.itertuples():
        ax.annotate(f"{row.sse / 1e6:.2f}M, down {row.sse_drop_share:.1%}", (row.k, row.sse), xytext=(7, 7), textcoords="offset points", fontsize=9.5, color=INK2)
    ax.set_ylabel("SSE (millions)")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value / 1e6:.1f}")
    ax.set(xlim=(1.6, 10.4), ylim=(1.0e6, 1.85e6))
    ax.set_title("SSE bends gently at k=3 and keeps falling, with no sharp knee", loc="left", pad=24)
    subtitle(ax, f"k=5 has the lowest required SSE, as a larger k always does. Percentages compare with k-1 (k=1: {(metrics['sse'].iloc[0] + metrics['sse_drop'].iloc[0]) / 1e6:.2f}M). The y-axis starts at 1.0M.")
    save(fig, "29_clustering_elbow.png")


def silhouette(metrics):
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    required = by_k(ax, metrics, "silhouette")
    for row in required.itertuples():
        ax.annotate(f"{row.silhouette:.3f}", (row.k, row.silhouette), xytext=(8, 8), textcoords="offset points", fontsize=10, color=INK2)
    ax.set_ylabel("Mean silhouette")
    ax.set(xlim=(1.6, 10.4), ylim=(0, 0.33))
    ax.set_title("Silhouette is highest at k=2 and weak at every k", loc="left", pad=24)
    subtitle(ax, f"On a fixed random sample of {metrics['silhouette_sample_size'].iloc[0]:,} rows. On five other samples, a value moves by at most {(metrics['silhouette_high'] - metrics['silhouette_low']).max():.3f}.")
    save(fig, "30_clustering_silhouette.png")


def k5_profile():
    read = dict(keep_default_na=False, na_values=[""])
    clusters = pd.read_csv(DATA / "clustering_events.csv", usecols=["event_id", "cluster_k5"], **read)
    events = pd.read_csv(DATA / "classification_events.csv", usecols=["event_id", "size_category", "exchange", "reaction_category"], **read)
    profile = pd.read_csv(DATA / "clustering_k5_profile.csv", **read).sort_values("mean_market_cap", ascending=False)
    merged = clusters.merge(events, on="event_id", validate="one_to_one")
    panels = [("size_category", SIZE_ORDER, "Company size"), ("exchange", EXCHANGES, "Exchange"), ("reaction_category", CLASSES, "Market reaction")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), gridspec_kw={"width_ratios": [len(p[1]) for p in panels]})
    for ax, (col, order, title) in zip(axes, panels):
        shares = pd.crosstab(merged["cluster_k5"], merged[col], normalize="index").reindex(index=profile["cluster"], columns=order, fill_value=0)
        image = ax.imshow(shares.to_numpy(), cmap=SHARES, vmin=0, vmax=1, aspect="auto")
        for row, values in enumerate(shares.to_numpy()):
            for column, value in enumerate(values):
                ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=10, color=SURFACE if value > 0.55 else INK)
        ax.set_xticks(range(len(order)), [c.replace(" ", "\n") for c in order], fontsize=9.5)
        ax.xaxis.tick_top()
        ax.set_yticks([])
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", pad=44, fontsize=12)
    axes[0].set_yticks(range(len(profile)), [f"Cluster {r.cluster}: {r.short_name}\n{r.events:,} events, mean cap {usd(r.mean_market_cap)}" for r in profile.itertuples()], fontsize=9.5)
    bar = fig.colorbar(image, ax=list(axes), orientation="horizontal", fraction=0.04, pad=0.04, aspect=50)
    bar.set_label("Share of the cluster's events (each row of each panel adds to 100%)")
    bar.ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    bar.outline.set_visible(False)
    fig.suptitle("The k=5 clusters split on company size and exchange, not on market reaction", x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.012, 0.915, "Clusters are ordered by mean market_cap. Size and exchange were inputs; market reaction was not. Its mix shifts toward Strong reactions at smaller companies, but no cluster is one reaction.", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.215, right=0.985, top=0.78, bottom=0.17, wspace=0.05)
    save(fig, "31_clustering_k5_profile.png", tight=False)


def between_clusters(values, clusters):
    """Share of a column's variance that lies between the clusters."""
    values = pd.Series(np.asarray(values, float))
    means = values.groupby(np.asarray(clusters)).transform("mean")
    return ((means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum()


def cluster_legend(fig, profile, colours, extra=()):
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=7, color=colours[r.cluster], label=f"Cluster {r.cluster}: {r.short_name}") for r in profile.itertuples()]
    handles.extend(extra)
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", fontsize=9.5)


def pca_by_cluster(events, profile, colours):
    """Chart 32: one point per company, using the company's most common k=5 cluster."""
    counts = pd.crosstab(events["ticker"], events["cluster_k5"])
    top = counts.max(axis=1)
    tied = counts.eq(top, axis=0).sum(axis=1) > 1
    company_cluster = counts.idxmax(axis=1).rename("cluster_k5")   # a tie goes to the lowest cluster number
    mixed = counts.gt(0).sum(axis=1) > 1
    scores = pd.read_csv(DATA / "pca_companies_scores.csv", keep_default_na=False, na_values=[""])
    scored = scores.merge(company_cluster, left_on="ticker", right_index=True, how="left", validate="one_to_one")
    assert scored["cluster_k5"].notna().all()
    covered = counts.index.isin(scores["ticker"])
    coverage_by_cluster = pd.Series(covered, index=counts.index).groupby(company_cluster).mean()

    event_scores = events[["ticker", "cluster_k5"]].merge(scores, on="ticker")
    event_shares = [between_clusters(event_scores[pc], event_scores["cluster_k5"]) for pc in ("PC1", "PC2")]
    company_shares = [between_clusters(scored[pc], scored["cluster_k5"]) for pc in ("PC1", "PC2")]
    medians = scored.groupby("cluster_k5")[["PC1", "PC2"]].median().reindex(profile["cluster"])
    larger = scored["cluster_k5"].isin(profile["cluster"].iloc[:3])
    pc2_larger = between_clusters(scored.loc[larger, "PC2"], scored.loc[larger, "cluster_k5"])
    pc2_smaller = between_clusters(scored.loc[~larger, "PC2"], scored.loc[~larger, "cluster_k5"])
    plane_silhouette = silhouette_score(scored[["PC1", "PC2"]], scored["cluster_k5"])
    variance = pd.read_csv(DATA / "pca_companies_explained_variance.csv")["explained_variance_ratio"] * 100

    rng = np.random.default_rng(4412)   # the same draw order as chart 24
    shuffled = scored.iloc[rng.permutation(len(scored))]
    point_colours = shuffled["cluster_k5"].map(colours)
    pc1_lim, pc2_lim = window(scored["PC1"]), window(scored["PC2"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, (xlim, ylim, name) in zip(axes, [(None, None, "All companies"), (pc1_lim, pc2_lim, "Zoomed to the middle 99%")]):
        ax.scatter(shuffled["PC1"], shuffled["PC2"], c=point_colours, s=15, linewidths=0.25, edgecolors=SURFACE)
        ax.set_xlabel(f"PC1 ({variance[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1f}% of variance)")
        if xlim:
            ax.set(xlim=xlim, ylim=ylim)
            outside = (~scored["PC1"].between(*xlim) | ~scored["PC2"].between(*ylim)).sum()
            subtitle(ax, f"{outside} of {len(scored):,} companies fall outside this window")
        else:
            subtitle(ax, f"Nothing hidden. {len(scored):,} of {len(counts):,} companies have a Task 1 score")
        ax.set_title(name, loc="left", pad=20)
    cluster_legend(fig, profile, colours)
    fig.suptitle(f"The clusters line up along PC1, and PC2 splits them among larger companies (silhouette {plane_silhouette:+.2f})", x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    save(fig, "32_clustering_pca_by_cluster.png", tight=False)
    return [
        "Chart 32: Task 1's processed/pca_companies_scores.csv joined by ticker to each company's most common cluster_k5, taken from processed/clustering_events.csv with ticker from classification_events.csv by event_id.",
        f"  {mixed.sum()} of {len(counts):,} companies have events in more than one cluster. {tied.sum()} of them needed a tie-break, {tied[covered].sum()} of those plotted; a tie goes to the lowest cluster number.",
        f"  {covered.sum():,} of {len(counts):,} companies ({covered.mean():.2%}) have a PC1/PC2 score and are plotted. The rest were left out of Task 1. Coverage by cluster: "
        + ", ".join(f"{c}: {coverage_by_cluster[c]:.1%}" for c in profile["cluster"]) + ".",
        f"  Variance between clusters across events: PC1 {event_shares[0]:.1%}, PC2 {event_shares[1]:.1%} (the 87.3% and 54.4% of Task 3). Across the plotted companies, one point each: PC1 {company_shares[0]:.1%}, PC2 {company_shares[1]:.1%}.",
        f"  PC2 variance between clusters {' , '.join(map(str, profile['cluster'].iloc[:3])).replace(' ,', ',')}: {pc2_larger:.1%}; between clusters {', '.join(map(str, profile['cluster'].iloc[3:]))}: {pc2_smaller:.1%}.",
        "  Median PC1, PC2 by cluster: " + ", ".join(f"{c}: {row.PC1:+.2f}, {row.PC2:+.2f}" for c, row in medians.iterrows()) + f". Silhouette in this plane: {plane_silhouette:+.2f}.",
    ]


def car_by_market_cap(events, profile, colours):
    """Chart 33: market_cap against the reaction, on the rows of the silhouette sample."""
    car = pd.read_csv(DATA / "full_dataset.csv", usecols=["event_id", "car_minus1_plus1"], keep_default_na=False, na_values=[""])
    merged = events.merge(car, on="event_id", how="left", validate="one_to_one")
    assert merged["car_minus1_plus1"].notna().all() and (merged["event_id"] == events["event_id"]).all()
    merged["car_percent"] = merged["car_minus1_plus1"] * 100
    sample = merged.iloc[np.random.RandomState(SAMPLE_SEED).permutation(len(merged))[:SAMPLE]]   # already in random order, so no cluster is always on top
    off_chart = sample["car_percent"].abs() > CAR_LIMIT
    bands = sample.groupby("cluster_k5")["car_percent"].quantile([0.1, 0.5, 0.9]).unstack().reindex(profile["cluster"])
    bands["off_chart"] = off_chart.groupby(sample["cluster_k5"]).sum()
    bands["market_cap"] = sample.groupby("cluster_k5")["market_cap"].median()
    inside = sample[~off_chart]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.scatter(sample["market_cap"], sample["car_percent"], c=sample["cluster_k5"].map(colours), s=6, linewidths=0, alpha=0.45, rasterized=True)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.vlines(bands["market_cap"], bands[0.1], bands[0.9], color=INK, linewidth=2)
    ax.scatter(bands["market_cap"], bands[0.5], s=46, color=INK, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    for cluster, row in bands.iterrows():
        ax.annotate(f"{cluster}", (row["market_cap"], row[0.9]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.2))
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(usd)
    ax.minorticks_off()
    ax.set_ylim(-CAR_LIMIT, CAR_LIMIT)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:+.0f}%" if value else "0%")
    ax.set_xlabel("market_cap (log scale)")
    ax.set_ylabel("CAR[-1,+1]")
    ax.set_title("All five clusters straddle zero, and the spread of CAR narrows as companies get larger", loc="left", pad=24)
    subtitle(ax, f"Fixed random sample of {SAMPLE:,} of {len(merged):,} events (random_state={SAMPLE_SEED}). {off_chart.sum()} sampled points lie beyond ±{CAR_LIMIT}% and are off the chart.")
    bars = plt.Line2D([], [], color=INK, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label="Numbered cluster's median, 10th-90th percentile")
    cluster_legend(fig, profile, colours, extra=[bars])
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, "33_clustering_market_cap_vs_car.png", tight=False)
    return [
        f"Chart 33: classification_events.csv (market_cap, ticker), clustering_events.csv (cluster_k5) and car_minus1_plus1 from full_dataset.csv, joined by event_id. market_cap is on a log axis.",
        f"  Sample: {SAMPLE:,} of {len(merged):,} events, numpy permutation with seed {SAMPLE_SEED}, the same rows as the silhouette sample. {off_chart.sum()} of them ({off_chart.mean():.2%}) lie beyond ±{CAR_LIMIT}% and are off the chart but still in the sample and its percentiles.",
        "  CAR[-1,+1] median and 10th-90th percentile by cluster: " + ", ".join(f"{c}: {r[0.5]:+.2f}% ({r[0.1]:+.1f}% to {r[0.9]:+.1f}%), {int(r['off_chart'])} off chart" for c, r in bands.iterrows()) + ".",
        f"  Variance of CAR between clusters: {between_clusters(inside['car_percent'], inside['cluster_k5']):.2%} of the points on the chart, {between_clusters(merged['car_percent'], merged['cluster_k5']):.2%} of all events. Of |CAR|, on the chart: {between_clusters(inside['car_percent'].abs(), inside['cluster_k5']):.2%}.",
    ]


def append_to_log(lines):
    """Replace this section of the log, so rerunning the charts does not repeat it."""
    path = DATA / "clustering_events_log.txt"
    kept = path.read_text(encoding="utf-8").split(f"\n{LOG_MARKER}")[0].rstrip()
    path.write_text(f"{kept}\n\n{LOG_MARKER}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    metrics = pd.read_csv(DATA / "clustering_metrics.csv")
    elbow(metrics)
    silhouette(metrics)
    k5_profile()
    read = dict(keep_default_na=False, na_values=[""])   # real tickers such as "NA" stay strings
    events = pd.read_csv(DATA / "classification_events.csv", usecols=["event_id", "ticker", "market_cap"], **read)
    clusters = pd.read_csv(DATA / "clustering_events.csv", usecols=["event_id", "cluster_k5"], **read)
    events = events.merge(clusters, on="event_id", how="left", validate="one_to_one")
    assert events["cluster_k5"].notna().all()
    profile = pd.read_csv(DATA / "clustering_k5_profile.csv", **read).sort_values("mean_market_cap", ascending=False)
    colours = dict(zip(profile["cluster"], CLUSTER_SLOTS))
    append_to_log([*pca_by_cluster(events, profile, colours), *car_by_market_cap(events, profile, colours)])


if __name__ == "__main__":
    main()
