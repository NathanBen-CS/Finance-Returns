"""
EECS 4412 Phase 2, Task 3 -- k-means on the Task 2 events, compared with the Task 2 classifiers.

  1. take processed/classification_events.csv unchanged and encode it with classification_events.encode(),
     with the z-score fitted on all rows (no held-out test exists in clustering)
  2. KMeans(random_state=42, n_init=10) for k=2..10: SSE and silhouette per k
  3. k=2 vs Up/Down, k=3 vs Negative/Neutral/Positive, k=5 vs reaction_category: contingency, purity, NMI, ARI
     k=4 has no matching label, so it gets SSE, silhouette and cluster sizes only
  4. profile the k=5 clusters and measure what the clusters do follow
  5. compare with Task 2 on the same test rows: give each cluster its training rows' most common label
  6. sensitivity checks on scaling and seeds (reported only, not in the main results)

Run with the project venv, after classification_events.py:  ..\\..\\.venv\\Scripts\\python.exe clustering_events.py
Writes, in processed/:  clustering_events.csv, clustering_metrics.csv, clustering_k5_profile.csv, clustering_events_log.txt
                        220227286.xlsx (sheets T3_Clustering_*; any other sheets are kept)
Charts come from ../charts2/clustering_charts.py, which reads the files above.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from classification_events import CLASSES, FEATURES, LABEL, LOG_COLUMNS, NOMINAL, ORDINAL, SEED, encode, percent, tree_model, write_blocks
from pca_companies import write_sheet

HERE = Path(__file__).resolve().parent
PROC = HERE / "processed"
WORKBOOK = PROC / "220227286.xlsx"

ALL_K = list(range(2, 11))
REQUIRED_K = [2, 3, 4, 5]
ASSIGNMENT_K = [3, 4, 5]   # what EECS4412--F26--P02.pdf names; k=2 is this project's addition
KEPT_DEPTH = 10            # the max_depth kept in Task 2
SILHOUETTE_SAMPLE = 20_000
SHUFFLES = 20

DIRECTION = "direction"
THREE_WAY = "reaction_3way"
TO_THREE_WAY = {"Strong Negative": "Negative", "Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive", "Strong Positive": "Positive"}
LABEL_OF_K = {2: DIRECTION, 3: THREE_WAY, 5: LABEL}
ORDER_OF_LABEL = {DIRECTION: ["Down", "Up"], THREE_WAY: ["Negative", "Neutral", "Positive"], LABEL: CLASSES}
NAME_OF_LABEL = {DIRECTION: "Up/Down", THREE_WAY: "Negative/Neutral/Positive", LABEL: "reaction_category"}
EXCHANGE_NAMES = {"NYQ": "NYSE", "NMS": "Nasdaq Global Select", "NGM": "Nasdaq Global Market", "NCM": "Nasdaq Capital Market",
                  "ASE": "NYSE American", "PNK": "OTC Pink"}
VOLATILITY_TIERS = ["Low", "Medium", "High", "Very High"]
OTHER_COLUMNS = ["size_category", "exchange", "sector", "event_category", "filing_day_of_week", "filing_timeliness", "volatility_tier", "strong_or_not"]

log_lines = []


def say(text=""):
    """Print a line and keep it for clustering_events_log.txt."""
    print(text, flush=True)
    log_lines.append(text)


def kmeans(x, k, seed=SEED):
    return KMeans(k, random_state=seed, n_init=10).fit(x)


def largest_first(labels):
    """Cluster numbers are arbitrary, so renumber them 1..k from the largest cluster down."""
    order = np.argsort(-np.bincount(labels), kind="stable")
    new_number = np.empty(len(order), int)
    new_number[order] = np.arange(1, len(order) + 1)
    return new_number[labels]


def alignment(truth, clusters):
    matrix = contingency_matrix(truth, clusters)
    return {"purity": matrix.max(axis=0).sum() / len(truth), "nmi": normalized_mutual_info_score(truth, clusters),
            "ari": adjusted_rand_score(truth, clusters)}


def shuffled_alignment(truth, clusters):
    """What the three scores look like when the labels are shuffled, so there is nothing to find."""
    rng = np.random.default_rng(SEED)
    codes = pd.factorize(truth)[0]
    return pd.DataFrame([alignment(rng.permutation(codes), clusters) for _ in range(SHUFFLES)]).mean()


def between_groups(values, groups):
    """Share of a column's variance that lies between the groups."""
    values = pd.Series(np.asarray(values, float))
    means = values.groupby(np.asarray(groups)).transform("mean")
    return ((means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum()


def contingency(clusters, truth, order):
    table = pd.crosstab(clusters, truth).reindex(columns=order).reset_index(drop=True)
    table.insert(0, "cluster", sorted(np.unique(clusters)))
    table["total"] = table[order].sum(axis=1)
    table["largest_class_share"] = table[order].max(axis=1) / table["total"]
    return table


def predict(model, x, order):
    """The most likely label, with ties going to the first in order, as Task 2's predictions() does."""
    columns = [list(model.classes_).index(c) for c in order]
    return np.array(order)[model.predict_proba(x)[:, columns].argmax(axis=1)]


def yardstick(x, y, order, clusters, train, test):
    """Tree, Naive Bayes and labelled clusters scored the same way on the same test rows."""
    tree = tree_model(KEPT_DEPTH).fit(x[train], y[train])
    bayes = GaussianNB().fit(x[train], y[train])
    majority = pd.Series(y[train]).groupby(clusters[train]).agg(lambda s: s.value_counts().idxmax())
    guesses = {"Decision tree": predict(tree, x[test], order), "Gaussian Naive Bayes": predict(bayes, x[test], order),
               "k-means, cluster majority": majority.reindex(clusters[test]).to_numpy()}
    baseline = pd.Series(y[test]).value_counts(normalize=True).max()
    rows = [{"approach": name, "largest_class_baseline": baseline, "test_accuracy": accuracy_score(y[test], guess),
             "points_above_baseline": (accuracy_score(y[test], guess) - baseline) * 100,
             "nmi": normalized_mutual_info_score(y[test], guess), "ari": adjusted_rand_score(y[test], guess)} for name, guess in guesses.items()]
    return rows, majority, guesses


def size_words(shares):
    top = shares.index[:2]
    if shares.iloc[0] >= 0.7:
        return str(top[0])
    low, high = sorted(top, key=list(ORDINAL["size_category"]).index)
    return f"{low}/{high}"


def exchange_words(shares):
    if shares.iloc[0] >= 0.7:
        return EXCHANGE_NAMES[shares.index[0]]
    nasdaq = shares[[c for c in shares.index if c in ("NMS", "NGM", "NCM")]].sum()
    return "several Nasdaq tiers" if nasdaq >= 0.7 else "mixed exchanges"


def profile(df, clusters):
    """One plain-terms row per cluster, in cluster order."""
    rows = []
    for number in sorted(np.unique(clusters)):
        part = df[clusters == number]
        shares = {col: part[col].value_counts(normalize=True) for col in ("sector", "size_category", "exchange", "event_category")}
        sectors = shares["sector"]
        size, exchange = size_words(shares["size_category"]), exchange_words(shares["exchange"])
        rows.append({
            "cluster": number, "short_name": f"{size}, {exchange}", "events": len(part), "share": len(part) / len(df), "companies": part["ticker"].nunique(),
            "description": f"{size} companies on {exchange}; largest sectors {sectors.index[0]} {sectors.iloc[0]:.1%} and {sectors.index[1]} {sectors.iloc[1]:.1%}; "
                           f"most common filing {shares['event_category'].index[0]} {shares['event_category'].iloc[0]:.1%}",
            "dominant_sectors": f"{sectors.index[0]} {sectors.iloc[0]:.1%}, {sectors.index[1]} {sectors.iloc[1]:.1%}",
            "dominant_size_category": shares["size_category"].index[0], "dominant_size_share": shares["size_category"].iloc[0],
            "dominant_exchange": shares["exchange"].index[0], "dominant_exchange_share": shares["exchange"].iloc[0],
            "mean_market_cap": part["market_cap"].mean(), "median_market_cap": part["market_cap"].median(),
            "dominant_event_category": shares["event_category"].index[0], "dominant_event_share": shares["event_category"].iloc[0],
            **{c: (part[LABEL] == c).mean() for c in CLASSES},
            "strong_share": part[LABEL].isin(["Strong Negative", "Strong Positive"]).mean(), "up_share": (part[DIRECTION] == "Up").mean(),
            "mean_car": part["car_minus1_plus1"].mean(), "median_car": part["car_minus1_plus1"].median(),
            "median_abs_car": part["car_minus1_plus1"].abs().median(), "mean_trailing_volatility": part["trailing_volatility_1y"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    say("EECS 4412 Phase 2, Task 3: k-means clustering of the Task 2 events")
    say("=" * 70)
    read = dict(keep_default_na=False, na_values=[""])   # real tickers such as "NA" stay strings
    df = pd.read_csv(PROC / "classification_events.csv", parse_dates=["trading_date", "filing_date"], **read)
    # never inputs: the CAR gives the Up/Down reference label; volatility is only compared with the clusters afterwards
    extra = pd.read_csv(PROC / "full_dataset.csv", usecols=["event_id", "car_minus1_plus1", "trailing_volatility_1y", "volatility_tier"], **read)
    df = df.merge(extra, on="event_id", how="left", validate="one_to_one")
    assert df["car_minus1_plus1"].notna().all() and df[FEATURES].notna().all().all() and df["volatility_tier"].isin(VOLATILITY_TIERS).all()

    # ---------------------------------------------------------------- 1. table and encoding
    say("\nSTEP 1: TABLE AND ENCODING")
    say(f"  Source: processed/classification_events.csv, unchanged. {len(df):,} events from {df['ticker'].nunique():,} companies, already filtered for price_data_unreliable and blank features.")
    y5 = df[LABEL].to_numpy()
    train_rows, test_rows = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y5, random_state=SEED)
    assert (df.loc[test_rows, "split"] == "test").all() and (df.loc[train_rows, "split"] == "train").all()
    encoded = encode(df, np.arange(len(df)))
    task2_encoded = encode(df, train_rows)
    assert list(encoded.columns) == list(task2_encoded.columns)
    assert len(df) == 392_038 and df["ticker"].nunique() == 3_703 and encoded.shape[1] == 33
    x, x_task2 = encoded.to_numpy(), task2_encoded.to_numpy()
    say(f"  Matches Task 2 exactly: {len(df):,} rows, {df['ticker'].nunique():,} companies, {encoded.shape[1]} encoded columns (the same names, in the same order, from the same encode()).")
    group_sizes = {col: df[col].nunique() for col in NOMINAL}
    count_note = (f"The brief expected 61 encoded columns. These 8 features give {encoded.shape[1]}, the same as Task 2: {len(ORDINAL)} ordinal + {len(LOG_COLUMNS)} log columns + "
                  + " + ".join(f"{n} {col}" for col, n in group_sizes.items()) + ". 61 cannot come from this feature set; nothing was added to reach it.")
    say(f"  {count_note}")
    scaler_note = "Deliberate difference from Task 2: the z-score of log_market_cap and log_employees is fitted on all rows, not on training rows. Clustering has no held-out test to protect, so this is not leakage."
    say(f"  {scaler_note}")
    for col in LOG_COLUMNS:
        logged = np.log1p(df[col].astype(float))
        say(f"    log_{col}: mean {logged.mean():.4f}, sd {logged.std(ddof=0):.4f} on all rows; mean {logged.iloc[train_rows].mean():.4f}, sd {logged.iloc[train_rows].std(ddof=0):.4f} on Task 2's training rows")
    say("  reaction_category, the CAR and the other labels are not inputs. They are compared with the clusters only after fitting.")
    variance = encoded.var(axis=0, ddof=0)
    groups = pd.Series(encoded.columns, index=encoded.columns).str.split("=").str[0]
    variance_share = (variance.groupby(groups).sum() / variance.sum()).sort_values(ascending=False)
    size_columns = ["size_category", "log_market_cap", "log_employees"]
    size_variance = variance_share[size_columns].sum()
    say("  Share of total variance per feature, which is what k-means sees: " + ", ".join(f"{name} {share:.1%}" for name, share in variance_share.items()))

    # ---------------------------------------------------------------- 2. labels and k
    say("\nSTEP 2: WHY THESE FOUR K VALUES")
    car = df["car_minus1_plus1"].to_numpy()
    df[DIRECTION] = np.where(car > 0, "Up", "Down")
    df[THREE_WAY] = df[LABEL].map(TO_THREE_WAY)
    df["strong_or_not"] = np.where(df[LABEL].str.startswith("Strong"), "Strong", "Not strong")
    reasons = [
        "k=2 is a pure direction split: Up = car_minus1_plus1 > 0, Down = car_minus1_plus1 <= 0, defined from the CAR's sign and independent of the 5-class bucket.",
        "k=3 is sign with magnitude dropped: Negative (Strong Negative + Negative), Neutral, Positive (Positive + Strong Positive).",
        "k=4 has no clean collapse of the 5-class label. It is only an interpolation point on the elbow and silhouette curves, so no label is forced onto it.",
        "k=5 is reaction_category as stored, which meets the assignment's rule that one k must equal the number of existing classes.",
        "k=6 to 10 only extend the elbow curve for context and are not part of the comparisons.",
        f"The assignment names k={', '.join(map(str, ASSIGNMENT_K))}. k=2 is added on top of that, for the direction question.",
    ]
    for reason in reasons:
        say(f"  {reason}")
    say("  Up/Down counts: " + ", ".join(percent((df[DIRECTION] == c).sum(), len(df)) + f" {c}" for c in ORDER_OF_LABEL[DIRECTION]) + f". {(car == 0).sum()} rows have a stored CAR of exactly 0 and count as Down.")
    say("  3-way counts: " + ", ".join(percent((df[THREE_WAY] == c).sum(), len(df)) + f" {c}" for c in ORDER_OF_LABEL[THREE_WAY]))
    neutral = df[LABEL] == "Neutral"
    say(f"  Up/Down is not a collapse of the 5 classes: the Neutral rows split {(df.loc[neutral, DIRECTION] == 'Down').sum():,} Down and {(df.loc[neutral, DIRECTION] == 'Up').sum():,} Up.")

    # ---------------------------------------------------------------- 3. k-means
    say("\nSTEP 3: K-MEANS, K=2 TO 10")
    say(f"  scikit-learn KMeans, random_state={SEED}, n_init=10. Clusters are renumbered 1..k from the largest down, because k-means numbers mean nothing.")
    say(f"  Silhouette needs every pairwise distance, so it is computed on a fixed random sample of {SILHOUETTE_SAMPLE:,} rows (random_state={SEED}). The range is the same score on five other samples (random_state 0-4).")
    total_ss = ((x - x.mean(axis=0)) ** 2).sum()
    say(f"  Total sum of squares (the SSE of k=1): {total_ss:,.0f}")
    say(f"  {'k':>3}{'SSE':>13}{'drop':>11}{'drop %':>9}{'explained':>11}{'silhouette':>12}{'range':>18}{'iterations':>12}")
    clusters, metric_rows = {}, []
    previous = total_ss
    for k in ALL_K:
        model = kmeans(x, k)
        clusters[k] = largest_first(model.labels_)
        score = silhouette_score(x, model.labels_, sample_size=SILHOUETTE_SAMPLE, random_state=SEED)
        others = [silhouette_score(x, model.labels_, sample_size=SILHOUETTE_SAMPLE, random_state=s) for s in range(5)]
        metric_rows.append({"k": k, "role": "required" if k in REQUIRED_K else "context", "sse": model.inertia_, "sse_drop": previous - model.inertia_,
                            "sse_drop_share": (previous - model.inertia_) / previous, "variance_explained": 1 - model.inertia_ / total_ss, "silhouette": score,
                            "silhouette_sample_size": SILHOUETTE_SAMPLE, "silhouette_low": min(others), "silhouette_high": max(others), "iterations": model.n_iter_})
        row = metric_rows[-1]
        say(f"  {k:>3}{row['sse']:>13,.0f}{row['sse_drop']:>11,.0f}{row['sse_drop_share']:>9.1%}{row['variance_explained']:>11.1%}{score:>12.4f}{f'{min(others):.4f}-{max(others):.4f}':>18}{model.n_iter_:>12}")
        previous = model.inertia_
    metrics = pd.DataFrame(metric_rows).set_index("k", drop=False)
    assert (metrics["iterations"] < 300).all()
    is_monotonic = metrics["sse"].is_monotonic_decreasing
    required = metrics.loc[REQUIRED_K]
    lowest_k = int(required["sse"].idxmin())
    drops = metrics["sse_drop_share"]
    sse_read = (f"Lowest SSE among the required k: k={lowest_k} ({metrics.loc[lowest_k, 'sse']:,.0f}), which is also the lowest of the assignment's k=3, 4, 5. "
                f"This is expected and does not make k=5 the best choice: SSE {'never rises' if is_monotonic else 'is not monotonic here'} as k grows, because more centroids always fit closer.")
    knee_read = (f"Elbow: there is a mild bend at k=3 but no sharp knee. On the k=2 to 10 curve SSE drops by {metrics.loc[3, 'sse_drop']:,.0f} reaching k=3, {metrics.loc[4, 'sse_drop']:,.0f} reaching k=4 and {metrics.loc[5, 'sse_drop']:,.0f} reaching k=5, "
                 f"then by {drops[6]:.1%}, {drops[7]:.1%}, {drops[8]:.1%}, {drops[9]:.1%} and {drops[10]:.1%} of what is left, with no step that stands out. Going from one cluster to k=2 removes {drops[2]:.1%} of the total sum of squares, and k=10 still explains only {metrics.loc[10, 'variance_explained']:.1%}.")
    silhouette_read = (f"Silhouette is highest at k=2 ({metrics.loc[2, 'silhouette']:.4f}) and falls as k grows, with k=4 ({metrics.loc[4, 'silhouette']:.4f}) and k=5 ({metrics.loc[5, 'silhouette']:.4f}) level within sampling noise. "
                       "Every value is weak: the rows form a gradient rather than separate groups.")
    say(f"  {sse_read}\n  {knee_read}\n  {silhouette_read}")
    say("\n  Cluster sizes:")
    size_rows = []
    for k in REQUIRED_K:
        for number in sorted(np.unique(clusters[k])):
            inside = clusters[k] == number
            size_rows.append({"k": k, "cluster": number, "events": inside.sum(), "share": inside.mean(), "companies": df.loc[inside, "ticker"].nunique()})
        say(f"    k={k}: " + ", ".join(f"{r['cluster']}: {percent(r['events'], len(df))}" for r in size_rows if r["k"] == k))
    cluster_sizes = pd.DataFrame(size_rows)

    # ---------------------------------------------------------------- 4. alignment
    say("\nSTEP 4: DO THE CLUSTERS RECOVER THE LABEL? (k=4 has no label and is not checked)")
    alignment_rows, tables = [], {}
    for k, label in LABEL_OF_K.items():
        truth = df[label].to_numpy()
        scores, chance = alignment(truth, clusters[k]), shuffled_alignment(truth, clusters[k])
        largest = df[label].value_counts(normalize=True).max()
        alignment_rows.append({"k": k, "label": NAME_OF_LABEL[label], **scores, "largest_class_share": largest, "purity_above_largest_class_points": (scores["purity"] - largest) * 100,
                               "purity_shuffled": chance["purity"], "nmi_shuffled": chance["nmi"], "ari_shuffled": chance["ari"]})
        tables[k] = contingency(clusters[k], truth, ORDER_OF_LABEL[label])
        say(f"\n  k={k} vs {NAME_OF_LABEL[label]}: purity {scores['purity']:.4f} (always guessing the largest class gives {largest:.4f}), NMI {scores['nmi']:.4f}, ARI {scores['ari']:.4f}. "
            f"With shuffled labels: purity {chance['purity']:.4f}, NMI {chance['nmi']:.5f}, ARI {chance['ari']:+.5f}.")
        say(tables[k].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    aligned = pd.DataFrame(alignment_rows).set_index("k", drop=False)

    # ---------------------------------------------------------------- 5. profile of k=5
    say("\nSTEP 5: THE k=5 CLUSTERS IN PLAIN TERMS")
    profiles = profile(df, clusters[5])
    for row in profiles.itertuples():
        say(f"  Cluster {row.cluster} ({percent(row.events, len(df))} from {row.companies:,} companies): {row.description}.")
        say(f"    mean market_cap ${row.mean_market_cap / 1e9:,.2f}B, median ${row.median_market_cap / 1e9:,.2f}B; dominant size_category {row.dominant_size_category} {row.dominant_size_share:.1%}; "
            f"dominant exchange {row.dominant_exchange} {row.dominant_exchange_share:.1%}")
        say("    " + ", ".join(f"{c} {profiles.loc[row.Index, c]:.1%}" for c in CLASSES))
        say(f"    Strong share {row.strong_share:.1%}; Up share {row.up_share:.1%}; mean CAR {row.mean_car:+.2%}, median CAR {row.median_car:+.2%}, median |CAR| {row.median_abs_car:.2%}; mean trailing volatility {row.mean_trailing_volatility:.2f} (reference only)")
    event_share = df["event_category"].value_counts(normalize=True)
    say(f"  Share of all events by event_category, for reference: " + ", ".join(f"{name} {share:.1%}" for name, share in event_share.head(4).items()))
    mega_floor = df.loc[df["size_category"] == "Mega-cap", "market_cap"].min()
    say(f"  size_category is quartiles of this dataset's companies: Mega-cap starts at ${mega_floor / 1e9:,.1f}B here, far below the conventional meaning of the word.")

    # ---------------------------------------------------------------- 6. what the clusters follow
    say("\nSTEP 6: WHAT THE CLUSTERS FOLLOW INSTEAD")
    say(f"  {'column':<24}" + "".join(f"{f'k={k} NMI':>10}{f'k={k} ARI':>10}" for k in LABEL_OF_K))
    other_rows = []
    for col in OTHER_COLUMNS:
        row = {"column": col}
        for k in LABEL_OF_K:
            row[f"k{k}_nmi"] = normalized_mutual_info_score(df[col], clusters[k])
            row[f"k{k}_ari"] = adjusted_rand_score(df[col], clusters[k])
        other_rows.append(row)
        say(f"  {col:<24}" + "".join(f"{row[f'k{k}_nmi']:>10.4f}{row[f'k{k}_ari']:>10.4f}" for k in LABEL_OF_K))
    other = pd.DataFrame(other_rows).set_index("column", drop=False)
    pca = pd.read_csv(PROC / "pca_companies_scores.csv", **read)
    with_scores = df[["ticker"]].merge(pca, on="ticker", how="left")
    scored = with_scores["PC1"].notna().to_numpy()
    consistency_rows = []
    for k in LABEL_OF_K:
        per_company = pd.Series(clusters[k]).groupby(df["ticker"].to_numpy())
        consistency_rows.append({"k": k, "companies_wholly_in_one_cluster": (per_company.nunique() == 1).mean(),
                                 "events_in_their_companys_main_cluster": per_company.agg(lambda s: s.value_counts().iloc[0]).sum() / len(df),
                                 **{f"{pc}_variance_between_clusters": between_groups(with_scores.loc[scored, pc], clusters[k][scored]) for pc in ("PC1", "PC2", "PC3")}})
    consistency = pd.DataFrame(consistency_rows).set_index("k", drop=False)
    say(f"  Task 1 scores cover {scored.mean():.1%} of these events ({with_scores.loc[scored, 'ticker'].nunique():,} companies).")
    for row in consistency.itertuples():
        say(f"  k={row.k}: {row.companies_wholly_in_one_cluster:.1%} of companies sit wholly in one cluster; {row.events_in_their_companys_main_cluster:.1%} of events are in their company's main cluster; "
            f"the clusters hold {row.PC1_variance_between_clusters:.1%} of the variance of Task 1's PC1, {row.PC2_variance_between_clusters:.1%} of PC2 and {row.PC3_variance_between_clusters:.1%} of PC3.")
    company_up = between_groups(df[DIRECTION] == "Up", df["ticker"])
    company_strong = between_groups(df["strong_or_not"] == "Strong", df["ticker"])
    company_chance = (df["ticker"].nunique() - 1) / (len(df) - 1)
    say(f"  Share of the Up/Down variance that lies between companies: {company_up:.2%}. For Strong vs not: {company_strong:.2%}. Random labels would give about {company_chance:.2%} from the number of companies alone.")
    say("  A grouping of companies can only reach what lies between companies, so this caps what these clusters can recover.")

    # ---------------------------------------------------------------- 7. comparison with Task 2
    say("\nSTEP 7: COMPARISON WITH TASK 2, SCORED THE SAME WAY")
    say("  k-means has no predictions, so each cluster gets the most common label among its training rows and predicts that for its test rows.")
    say(f"  Tree (max_depth={KEPT_DEPTH}) and GaussianNB are refitted on each version of the label, with Task 2's encoding and split. k-means itself saw no labels; it did see the test rows' features.")
    by_company_train, by_company_test = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(x_task2, y5, df["ticker"]))
    cutoff = df["trading_date"].quantile(0.8)
    early, late = np.flatnonzero(df["trading_date"] < cutoff), np.flatnonzero(df["trading_date"] >= cutoff)
    splits = {"Task 2 split (stratified 80/20)": (train_rows, test_rows), "No company in both train and test": (by_company_train, by_company_test),
              f"Train on the earliest 80%, test from {cutoff.date()}": (early, late)}
    board_rows, majorities, guessed = [], {}, {}
    say(f"  {'split':<46}{'label':<28}{'approach':<28}{'baseline':>9}{'accuracy':>10}{'above':>8}{'NMI':>9}{'ARI':>9}")
    for split, (train, test) in splits.items():
        for k, label in LABEL_OF_K.items():
            rows, majority, guesses = yardstick(x_task2, df[label].to_numpy(), ORDER_OF_LABEL[label], clusters[k], train, test)
            majorities[split, k], guessed[split, k] = majority, guesses
            for row in rows:
                row = {"split": split, "label": NAME_OF_LABEL[label], "k": k, **row}
                board_rows.append(row)
                say(f"  {split:<46}{NAME_OF_LABEL[label]:<28}{row['approach']:<28}{row['largest_class_baseline']:>9.2%}{row['test_accuracy']:>10.2%}{row['points_above_baseline']:>8.2f}{row['nmi']:>9.4f}{row['ari']:>9.4f}")
    board = pd.DataFrame(board_rows)

    def scored_as(split, k, approach):
        return board[(board["split"] == split) & (board["k"] == k) & (board["approach"] == approach)].iloc[0]

    main_split, company_split, time_split = splits
    tree_name, bayes_name, cluster_name = "Decision tree", "Gaussian Naive Bayes", "k-means, cluster majority"
    task2 = pd.read_csv(PROC / "classification_metrics.csv")
    task2 = task2[task2["metric"] == "test_accuracy"].set_index("model")["value"]
    assert np.isclose(scored_as(main_split, 5, tree_name)["test_accuracy"], task2[f"Decision Tree (max_depth={KEPT_DEPTH})"], atol=1e-12)
    assert np.isclose(scored_as(main_split, 5, bayes_name)["test_accuracy"], task2["Gaussian Naive Bayes"], atol=1e-12)
    say(f"  Check: the tree and Naive Bayes on the 5-class label reproduce Task 2's accuracy exactly ({task2.iloc[0]:.4%} and {task2.iloc[1]:.4%}).")
    for k in LABEL_OF_K:
        said = majorities[main_split, k]
        missing = [c for c in ORDER_OF_LABEL[LABEL_OF_K[k]] if c not in set(said)]
        say(f"  k={k} cluster majorities: " + ", ".join(f"{number}: {name}" for number, name in said.items()) + (f". Never predicted: {', '.join(missing)}." if missing else "."))

    # ---------------------------------------------------------------- 8. sensitivity
    say("\nSTEP 8: CHECKS ON SCALING AND SEEDS (reported only; the results above use the requested encoding and seed)")
    ordinal_z = encoded.copy()
    ordinal_z[list(ORDINAL)] = StandardScaler().fit_transform(ordinal_z[list(ORDINAL)])
    variants = {
        "z-score fitted on Task 2's training rows": x_task2,
        "Ordinal columns z-scored too": ordinal_z.to_numpy(),
        "market_cap as the only size column": encode(df, np.arange(len(df)), ordinal={"filing_timeliness": ORDINAL["filing_timeliness"]}, log_columns=["market_cap"]).to_numpy(),
        "No size columns": encode(df, np.arange(len(df)), ordinal={"filing_timeliness": ORDINAL["filing_timeliness"]}, log_columns=[]).to_numpy(),
    }
    variant_rows = []
    say(f"  {'check':<44}{'k':>3}{'purity':>9}{'NMI':>9}{'ARI':>9}{'ARI with main':>15}{'NMI size':>10}{'NMI exch':>10}{'NMI event':>11}")
    for name, features in variants.items():
        for k, label in LABEL_OF_K.items():
            found = kmeans(features, k).labels_
            row = {"check": name, "k": k, **alignment(df[label].to_numpy(), found), "ari_with_main_clusters": adjusted_rand_score(clusters[k], found),
                   "nmi_size_category": normalized_mutual_info_score(df["size_category"], found), "nmi_exchange": normalized_mutual_info_score(df["exchange"], found),
                   "nmi_event_category": normalized_mutual_info_score(df["event_category"], found)}
            variant_rows.append(row)
            say(f"  {name:<44}{k:>3}{row['purity']:>9.4f}{row['nmi']:>9.4f}{row['ari']:>9.4f}{row['ari_with_main_clusters']:>15.3f}{row['nmi_size_category']:>10.3f}{row['nmi_exchange']:>10.3f}{row['nmi_event_category']:>11.3f}")
    scale_checks = pd.DataFrame(variant_rows)
    seed_rows = []
    say("  Five other seeds (random_state 0-4):")
    for k in REQUIRED_K:
        runs = [kmeans(x, k, seed=s) for s in range(5)]
        with_main = [adjusted_rand_score(clusters[k], run.labels_) for run in runs]
        sse = [run.inertia_ for run in runs]
        row = {"k": k, "sse_low": min(sse), "sse_high": max(sse), "sse_main": metrics.loc[k, "sse"], "ari_with_main_low": min(with_main), "ari_with_main_high": max(with_main)}
        if k in LABEL_OF_K:
            with_label = [adjusted_rand_score(df[LABEL_OF_K[k]], run.labels_) for run in runs]
            row.update({"ari_with_label_low": min(with_label), "ari_with_label_high": max(with_label)})
        seed_rows.append(row)
        say(f"    k={k}: SSE {min(sse):,.0f}-{max(sse):,.0f} (main {metrics.loc[k, 'sse']:,.0f}); ARI with the main clusters {min(with_main):.3f}-{max(with_main):.3f}"
            + (f"; ARI with the label {min(with_label):+.4f} to {max(with_label):+.4f}" if k in LABEL_OF_K else ""))
    seed_checks = pd.DataFrame(seed_rows).set_index("k", drop=False)

    say("  Volatility is not an input, but does it explain the magnitude pattern? Strong share by k=5 cluster, inside each volatility tier:")
    within_tier = (df["strong_or_not"] == "Strong").groupby([df["volatility_tier"], clusters[5]]).mean().unstack().reindex(VOLATILITY_TIERS)
    tier_spread = within_tier.max(axis=1) - within_tier.min(axis=1)
    overall_spread = profiles["strong_share"].max() - profiles["strong_share"].min()
    say(within_tier.assign(spread=tier_spread).to_string(float_format=lambda v: f"{v:.3f}"))
    say(f"  Spread between clusters: {overall_spread * 100:.1f} points overall, {tier_spread.min() * 100:.1f} to {tier_spread.max() * 100:.1f} points inside one tier.")
    guesses_5 = guessed[main_split, 5]
    tiers_of_test, truth_5 = df["volatility_tier"].to_numpy()[test_rows], y5[test_rows]
    tier_rows = []
    for tier in VOLATILITY_TIERS:
        inside = tiers_of_test == tier
        baseline = pd.Series(truth_5[inside]).value_counts(normalize=True).max()
        tier_rows.append({"volatility_tier": tier, "test_rows": inside.sum(), "largest_class_baseline": baseline,
                          "tree_accuracy": accuracy_score(truth_5[inside], guesses_5[tree_name][inside]), "clusters_accuracy": accuracy_score(truth_5[inside], guesses_5[cluster_name][inside]),
                          "tree_points_above": (accuracy_score(truth_5[inside], guesses_5[tree_name][inside]) - baseline) * 100,
                          "clusters_points_above": (accuracy_score(truth_5[inside], guesses_5[cluster_name][inside]) - baseline) * 100})
        row = tier_rows[-1]
        say(f"    {tier:<10} 5-class test rows {row['test_rows']:>7,}: baseline {baseline:.2%}, tree {row['tree_accuracy']:.2%} ({row['tree_points_above']:+.2f}), labelled clusters {row['clusters_accuracy']:.2%} ({row['clusters_points_above']:+.2f})")
    tier_checks = pd.DataFrame(tier_rows)

    guesses_2 = guessed[main_split, 2]
    tree_2, clusters_2, truth_2 = guesses_2[tree_name], guesses_2[cluster_name], df[DIRECTION].to_numpy()[test_rows]
    agree_2 = (tree_2 == clusters_2).mean()
    differ_2 = tree_2 != clusters_2
    tree_right_when_differ = (tree_2[differ_2] == truth_2[differ_2]).mean()
    tree_up_by_size = pd.Series(tree_2 == "Up").groupby(df["size_category"].to_numpy()[test_rows]).mean()
    up_by_size = (df[DIRECTION] == "Up").groupby(df["size_category"]).mean()
    bayes_neutral, true_neutral = (guessed[main_split, 3][bayes_name] == "Neutral").mean(), (df[THREE_WAY].to_numpy()[test_rows] == "Neutral").mean()
    say(f"  Up/Down on the Task 2 test rows: tree and labelled clusters agree on {agree_2:.2%}; where they differ the tree is right {tree_right_when_differ:.2%} of the time. The tree says Up for {tree_up_by_size['Small']:.1%} of Small and {tree_up_by_size['Mega-cap']:.1%} of Mega-cap rows.")

    # ---------------------------------------------------------------- 9. comparison and verdict, in words
    by_size = profiles.sort_values("mean_market_cap")
    k2_up, k3_table = pd.crosstab(clusters[2], df[DIRECTION], normalize="index")["Up"], pd.crosstab(clusters[3], df[THREE_WAY], normalize="index")
    k5_tree, k5_bayes, k5_cluster = (scored_as(main_split, 5, name) for name in (tree_name, bayes_name, cluster_name))
    k3_tree, k3_bayes, k3_cluster = (scored_as(main_split, 3, name) for name in (tree_name, bayes_name, cluster_name))
    k2_tree, k2_bayes, k2_cluster = (scored_as(main_split, 2, name) for name in (tree_name, bayes_name, cluster_name))
    time_tree, time_bayes, time_cluster = (scored_as(time_split, 5, name) for name in (tree_name, bayes_name, cluster_name))
    time2_tree, time2_bayes, time2_cluster = (scored_as(time_split, 2, name) for name in (tree_name, bayes_name, cluster_name))
    company_tree, company_bayes, company_cluster = (scored_as(company_split, 5, name) for name in (tree_name, bayes_name, cluster_name))
    assert k5_tree["test_accuracy"] > k5_bayes["test_accuracy"] > k5_cluster["test_accuracy"] > k5_cluster["largest_class_baseline"]
    assert k3_tree["test_accuracy"] > k3_cluster["test_accuracy"] and k5_tree["nmi"] > k5_cluster["nmi"] and k5_tree["ari"] > k5_cluster["ari"]
    assert abs(k2_tree["test_accuracy"] - k2_cluster["test_accuracy"]) < 0.001
    assert (profiles["dominant_event_category"] == "Financial/Earnings").all()
    assert time_tree["points_above_baseline"] > time_cluster["points_above_baseline"] and company_tree["points_above_baseline"] > company_cluster["points_above_baseline"]
    assert (board[(board["split"] == time_split) & (board["k"] == 2)]["points_above_baseline"] < 0).all()
    comparison = [
        f"k=2 vs Up/Down: not recovered. Purity {aligned.loc[2, 'purity']:.4f} is {aligned.loc[2, 'purity_above_largest_class_points']:.2f} points above always guessing Down, NMI {aligned.loc[2, 'nmi']:.4f}, ARI {aligned.loc[2, 'ari']:.4f}. "
        f"The clusters split companies by size (NMI with size_category {other.loc['size_category', 'k2_nmi']:.3f}, ARI {other.loc['size_category', 'k2_ari']:.3f}), and Up share is {k2_up.iloc[0]:.1%} and {k2_up.iloc[1]:.1%}.",
        f"k=3 vs Negative/Neutral/Positive: not recovered. Purity {aligned.loc[3, 'purity']:.4f} is {aligned.loc[3, 'purity_above_largest_class_points']:.2f} points above the largest class, NMI {aligned.loc[3, 'nmi']:.4f}, ARI {aligned.loc[3, 'ari']:.4f}. "
        f"Negative share is {k3_table['Negative'].min():.1%} to {k3_table['Negative'].max():.1%} and Positive {k3_table['Positive'].min():.1%} to {k3_table['Positive'].max():.1%} across clusters; Neutral runs {k3_table['Neutral'].min():.1%} to {k3_table['Neutral'].max():.1%}. "
        f"NMI with size_category is {other.loc['size_category', 'k3_nmi']:.3f}.",
        f"k=5 vs reaction_category: the highest alignment of the three and still weak. Purity {aligned.loc[5, 'purity']:.4f} is {aligned.loc[5, 'purity_above_largest_class_points']:.2f} points above the largest class, NMI {aligned.loc[5, 'nmi']:.4f}, ARI {aligned.loc[5, 'ari']:.4f}. "
        f"The small part it finds is magnitude: Strong share runs from {by_size['strong_share'].iloc[-1]:.1%} in the cluster with the largest companies to {by_size['strong_share'].iloc[0]:.1%} in the cluster with the smallest, while Up share only runs {profiles['up_share'].min():.1%} to {profiles['up_share'].max():.1%}. "
        f"NMI with Strong vs not is {other.loc['strong_or_not', 'k5_nmi']:.4f} and with Up/Down {normalized_mutual_info_score(df[DIRECTION], clusters[5]):.4f}.",
        f"The axis the clusters follow is company size and exchange, as in Task 1. NMI with size_category is {other.loc['size_category', 'k2_nmi']:.3f}, {other.loc['size_category', 'k3_nmi']:.3f}, {other.loc['size_category', 'k5_nmi']:.3f} at k=2, 3, 5 and with exchange {other.loc['exchange', 'k2_nmi']:.3f}, {other.loc['exchange', 'k3_nmi']:.3f}, {other.loc['exchange', 'k5_nmi']:.3f}, against {aligned.loc[2, 'nmi']:.4f}, {aligned.loc[3, 'nmi']:.4f}, {aligned.loc[5, 'nmi']:.4f} with the matching label. "
        f"Task 1's PC1 is company size and its PC2 is mainly NMS vs NYQ. On the {scored.mean():.1%} of events with a Task 1 score, the clusters hold {consistency.loc[2, 'PC1_variance_between_clusters']:.1%}, {consistency.loc[3, 'PC1_variance_between_clusters']:.1%} and {consistency.loc[5, 'PC1_variance_between_clusters']:.1%} of the variance of PC1, and at k=5 also {consistency.loc[5, 'PC2_variance_between_clusters']:.1%} of PC2. "
        f"They group companies, not filings: at k=5, {consistency.loc[5, 'companies_wholly_in_one_cluster']:.1%} of companies sit wholly in one cluster, NMI with event_category is only {other.loc['event_category', 'k5_nmi']:.4f}, and Financial/Earnings is the most common filing in all five clusters, so a name like \"large-cap tech earnings filers\" would claim more than the clusters show.",
        f"Why they cannot do better: only {company_up:.2%} of the Up/Down variance lies between companies, and {company_chance:.2%} would appear by chance. Direction changes from filing to filing within a company, and these clusters do not.",
        f"Same test rows, 5-class label: tree {k5_tree['test_accuracy']:.2%}, Naive Bayes {k5_bayes['test_accuracy']:.2%}, labelled clusters {k5_cluster['test_accuracy']:.2%}, baseline {k5_tree['largest_class_baseline']:.2%}. "
        f"NMI {k5_tree['nmi']:.4f}, {k5_bayes['nmi']:.4f}, {k5_cluster['nmi']:.4f}; ARI {k5_tree['ari']:.4f}, {k5_bayes['ari']:.4f}, {k5_cluster['ari']:.4f}.",
        f"Same test rows, 3-way label: tree {k3_tree['test_accuracy']:.2%}, Naive Bayes {k3_bayes['test_accuracy']:.2%}, labelled clusters {k3_cluster['test_accuracy']:.2%}, baseline {k3_tree['largest_class_baseline']:.2%}. "
        f"The clusters have the higher NMI and ARI here ({k3_cluster['nmi']:.4f} and {k3_cluster['ari']:.4f}, tree {k3_tree['nmi']:.4f} and {k3_tree['ari']:.4f}), but all of these are close to zero.",
        f"Same test rows, Up/Down: k=2 behaves differently. Tree {k2_tree['test_accuracy']:.2%}, Naive Bayes {k2_bayes['test_accuracy']:.2%}, labelled clusters {k2_cluster['test_accuracy']:.2%}, baseline {k2_tree['largest_class_baseline']:.2%}. "
        f"The clusters tie the tree because neither finds direction. Both lean on size: Up share runs from {up_by_size['Small']:.1%} at Small to {up_by_size['Mega-cap']:.1%} at Mega-cap, and the tree says Up for {tree_up_by_size['Small']:.1%} of Small and {tree_up_by_size['Mega-cap']:.1%} of Mega-cap rows. "
        f"They agree on {agree_2:.1%} of test rows, and where they differ the tree is right {tree_right_when_differ:.1%} of the time.",
        f"Verdict: classification is the better approach for predicting reaction_category when a label is available, and that rests on the decision tree. On the 5-class label the tree gains {k5_tree['points_above_baseline']:.2f} points over the baseline against {k5_cluster['points_above_baseline']:.2f} for labelled clusters, and it stays ahead on unseen companies ({company_tree['points_above_baseline']:+.2f} vs {company_cluster['points_above_baseline']:+.2f}) and on the later rows ({time_tree['points_above_baseline']:+.2f} vs {time_cluster['points_above_baseline']:+.2f}). The clusters also need those same labels to predict anything. "
        f"Naive Bayes does not share this clearly: {k5_bayes['points_above_baseline']:+.2f} vs {k5_cluster['points_above_baseline']:+.2f} for labelled clusters on the Task 2 split, level on unseen companies ({company_bayes['points_above_baseline']:+.2f} vs {company_cluster['points_above_baseline']:+.2f}) and on the later rows ({time_bayes['points_above_baseline']:+.2f} vs {time_cluster['points_above_baseline']:+.2f}), and {abs(k3_bayes['points_above_baseline']):.2f} points below the baseline on the 3-way label, where it predicts Neutral for {bayes_neutral:.1%} of rows against {true_neutral:.1%} in truth. "
        "The tree's win is about how large a reaction is, not which way it goes. On direction no approach is better and none is useful: the gain is under half a point on the Task 2 split, and all three fall below the baseline on the later rows. "
        "Clustering is better described as surfacing company structure (size and exchange) than reaction structure.",
    ]
    say("\nSTEP 9: COMPARISON AND VERDICT")
    say("  " + "\n  ".join(comparison))

    # ---------------------------------------------------------------- 10. flags
    variant = scale_checks.set_index(["check", "k"])
    z_variant, cap_variant, no_size = "Ordinal columns z-scored too", "market_cap as the only size column", "No size columns"
    timing_nmi = max(other.loc[col, f"k{k}_nmi"] for col in ("filing_timeliness", "filing_day_of_week") for k in LABEL_OF_K)
    worst_seed = seed_checks.loc[5]
    same_day = df.groupby(["ticker", "trading_date"])["event_id"].transform("size") > 1
    assert df.groupby(["ticker", "trading_date"])[DIRECTION].nunique().max() == 1
    best_label_ari = scale_checks["ari"].abs().max()
    flags = [
        count_note,
        f"The size of each column matters here, which it did not in Task 2. Feature scale barely affects trees and Naive Bayes, but it drives k-means. With the requested encoding, size_category, log_market_cap and log_employees hold {size_variance:.1%} of the total variance, and size_category alone holds {variance_share['size_category']:.1%} because it stays on a 1-4 scale. "
        f"So size is what k-means mostly sees. Those columns also carry nearly one fact three times, as size_category is quartiles of market_cap. The encoding is kept as requested.",
        f"The cluster boundaries depend on that scaling, but the finding does not. Z-scoring the ordinals changes the partition a lot at k=3 and k=5 (ARI with the main clusters {variant.loc[(z_variant, 3), 'ari_with_main_clusters']:.3f} and {variant.loc[(z_variant, 5), 'ari_with_main_clusters']:.3f}), and removing the size columns moves the clusters onto exchange and event_category (at k=5, NMI {variant.loc[(no_size, 5), 'nmi_exchange']:.3f} and {variant.loc[(no_size, 5), 'nmi_event_category']:.3f}). "
        f"Across all these checks the largest ARI with the matching label is {best_label_ari:.4f}. Describe the clusters as what this encoding produces, not as natural groups. Fitting the z-score on Task 2's training rows instead changes almost nothing (ARI with the main clusters {variant.loc[('z-score fitted on Task 2' + chr(39) + 's training rows', 5), 'ari_with_main_clusters']:.3f} at k=5).",
        f"Rows from one company repeat one position on five of the eight features, because sector, exchange, size_category, market_cap and employees are per-company values. Only event_category, filing_day_of_week and filing_timeliness can move a filing, and they carry {variance_share[['event_category', 'filing_day_of_week', 'filing_timeliness']].sum():.1%} of the variance. Companies that file more count more.",
        f"Snapshot limitation: size_category, market_cap and employees are single October 2026 values attached to events back to {df['trading_date'].min().date()}, and sector and exchange are snapshots too, so cluster membership uses information from after most events.",
        f"The only direction signal may be an artefact. Up share is {by_size['up_share'].iloc[0]:.1%} in the cluster with the smallest companies and {by_size['up_share'].iloc[-1]:.1%} in the one with the largest. Companies that are small today may be small because their stock fell, and that cluster's mean CAR is {by_size['mean_car'].iloc[0]:+.2%} with a median of {by_size['median_car'].iloc[0]:+.2%}, so skewed returns would also produce this. "
        "Without point-in-time market caps, this data cannot separate those from a real effect. Do not read it as a way to predict direction.",
        f"Size stands in for the excluded volatility, and this accounts for most of the magnitude pattern. trailing_volatility_1y is not an input, but the k=5 clusters line up with its quartiles (NMI {other.loc['volatility_tier', 'k5_nmi']:.3f}), with mean volatility running {profiles['mean_trailing_volatility'].min():.2f} to {profiles['mean_trailing_volatility'].max():.2f}. "
        f"The Strong share differs by {overall_spread * 100:.1f} points between clusters overall but only {tier_spread.min() * 100:.1f} to {tier_spread.max() * 100:.1f} points inside one volatility tier. Inside a tier, on the 5-class test rows, the tree gains {tier_checks['tree_points_above'].min():+.2f} to {tier_checks['tree_points_above'].max():+.2f} points over that tier's largest class and labelled clusters {tier_checks['clusters_points_above'].min():+.2f} to {tier_checks['clusters_points_above'].max():+.2f}. "
        "So excluding the column did not fully remove what it measures.",
        f"Purity has a floor: it cannot fall far below the largest class's share, so read it against that share and not on its own. NMI and ARI are clearly above their shuffled values, but with {len(df):,} rows that shows little. The size of the effect is the point, and these rows are not independent: {percent(same_day.sum(), len(df))} share a company and trading day with another row, and always share its CAR.",
        f"Labelling clusters by majority is a yardstick, not a fair test of clustering, which never tries to predict anything. It also predicts few classes: at k=5 only {', '.join(sorted(set(majorities[main_split, 5])))}, and at k=3 never Neutral. Several k=5 majorities are near ties.",
        f"The random split flatters every approach. Training on the earliest 80% and testing from {cutoff.date()}, the 5-class baseline is {time_tree['largest_class_baseline']:.2%}: tree {time_tree['test_accuracy']:.2%}, Naive Bayes {time_bayes['test_accuracy']:.2%}, labelled clusters {time_cluster['test_accuracy']:.2%}. On Up/Down the baseline is {time2_tree['largest_class_baseline']:.2%}: tree {time2_tree['test_accuracy']:.2%}, Naive Bayes {time2_bayes['test_accuracy']:.2%}, labelled clusters {time2_cluster['test_accuracy']:.2%}. "
        f"On companies not seen in training, 5-class: tree {company_tree['test_accuracy']:.2%}, labelled clusters {company_cluster['test_accuracy']:.2%}, baseline {company_tree['largest_class_baseline']:.2%}.",
        f"filing_timeliness and filing_day_of_week can come after the reaction, as flagged in Task 2. This matters little here: their NMI with the clusters is at most {timing_nmi:.4f}.",
        f"k-means can settle in different local solutions at k=5. Over five other seeds SSE runs {worst_seed['sse_low']:,.0f} to {worst_seed['sse_high']:,.0f} (main {worst_seed['sse_main']:,.0f}) and ARI with the main clusters runs {worst_seed['ari_with_main_low']:.3f} to {worst_seed['ari_with_main_high']:.3f}, so one seed finds a visibly different partition with a slightly worse SSE. ARI with the label stays at {worst_seed['ari_with_label_low']:.4f} to {worst_seed['ari_with_label_high']:.4f}. No seed was chosen for its result.",
        f"Silhouette uses a {SILHOUETTE_SAMPLE:,}-row sample, not all rows. Over five other samples it moves by at most {(metrics['silhouette_high'] - metrics['silhouette_low']).max():.4f}, so k=4 and k=5 cannot be ranked. Mixing one-hot and numeric columns in one Euclidean distance also limits how much the score can mean.",
        f"Inherited from Task 2: {(car == 0).sum()} rows with a stored CAR of exactly 0 count as Down, one-way price suspects can still affect labels (not cluster inputs), blank employees removed more small companies, and the universe has survivorship bias.",
    ]
    say("\nFLAGS")
    for i, flag in enumerate(flags, start=1):
        say(f"  {i:>2}. {flag}")

    # ---------------------------------------------------------------- write everything
    assigned = df[["event_id"]].copy()
    for k in REQUIRED_K:
        assigned[f"cluster_k{k}"] = clusters[k]
    assigned[LABEL] = df[LABEL]
    assigned.assign(**{DIRECTION: df[DIRECTION], THREE_WAY: df[THREE_WAY]}).to_csv(PROC / "clustering_events.csv", index=False)
    metrics.to_csv(PROC / "clustering_metrics.csv", index=False)
    profiles.to_csv(PROC / "clustering_k5_profile.csv", index=False)

    share, count, score = "0.00%", "#,##0", "0.0000"
    fmt = {c: share for c in ("share", "sse_drop_share", "variance_explained", "largest_class_share", "largest_class_baseline", "test_accuracy", "dominant_size_share", "dominant_exchange_share",
                              "dominant_event_share", *CLASSES, "strong_share", "up_share", "mean_car", "median_car", "median_abs_car", "companies_wholly_in_one_cluster",
                              "events_in_their_companys_main_cluster", "PC1_variance_between_clusters", "PC2_variance_between_clusters", "PC3_variance_between_clusters")}
    fmt.update({c: score for c in ("silhouette", "silhouette_low", "silhouette_high", "purity", "nmi", "ari", "purity_shuffled", "nmi_shuffled", "ari_shuffled", "ari_with_main_clusters", "nmi_size_category",
                                   "nmi_exchange", "nmi_event_category", "ari_with_main_low", "ari_with_main_high", "ari_with_label_low", "ari_with_label_high",
                                   *[f"k{k}_{m}" for k in LABEL_OF_K for m in ("nmi", "ari")])})
    fmt.update({c: count for c in ("sse", "sse_drop", "events", "companies", "total", "silhouette_sample_size", "sse_low", "sse_high", "sse_main", "mean_market_cap", "median_market_cap")})
    fmt.update({"purity_above_largest_class_points": "0.00", "points_above_baseline": "0.00", "mean_trailing_volatility": "0.00"})
    contingency_formats = {c: count for labels in ORDER_OF_LABEL.values() for c in labels} | {"total": count, "largest_class_share": share}

    wb = load_workbook(WORKBOOK) if WORKBOOK.exists() else Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    write_sheet(wb, "T3_Clustering_Data", assigned, {f"cluster_k{k}": "0" for k in REQUIRED_K}, {"event_id": 12, "reaction_category": 19})
    write_blocks(wb, "T3_Clustering_Results", [
        (f"SSE and silhouette per k (silhouette on a fixed sample of {SILHOUETTE_SAMPLE:,} rows, random_state={SEED}; low and high are five other samples)", metrics, fmt),
        ("Cluster sizes, k=2, 3, 4, 5", cluster_sizes, fmt),
        ("k=2 vs Up/Down (Up = car_minus1_plus1 > 0)", tables[2], contingency_formats),
        ("k=3 vs Negative/Neutral/Positive", tables[3], contingency_formats),
        ("k=5 vs reaction_category", tables[5], contingency_formats),
        ("Purity, NMI and ARI against the matching label (k=4 has none)", aligned, fmt),
        ("k=5 cluster profiles (plain-terms descriptions are in T3_Clustering_Notes)", profiles[["cluster", "dominant_sectors", "short_name", *[c for c in profiles.columns if c not in ("cluster", "dominant_sectors", "short_name", "description")]]], fmt),
        ("What the clusters follow: NMI and ARI against other columns (volatility_tier and strong_or_not are not inputs)", other, fmt),
        ("Do the clusters group companies? And how they line up with Task 1's principal components", consistency, fmt),
        ("Comparison with Task 2, scored the same way on the same test rows", board, fmt),
        ("Sensitivity: scaling (not the main results)", scale_checks, fmt),
        ("Sensitivity: five other seeds (not the main results)", seed_checks, fmt),
        ("Strong share by k=5 cluster inside each volatility tier (volatility is not an input)", within_tier.assign(spread=tier_spread).reset_index().rename(columns={c: f"cluster {c}" for c in within_tier.columns}),
         {f"cluster {c}": share for c in within_tier.columns} | {"spread": share}),
        ("5-class accuracy inside each volatility tier, Task 2 test rows", tier_checks, fmt | {"tree_accuracy": share, "clusters_accuracy": share, "tree_points_above": "0.00", "clusters_points_above": "0.00", "test_rows": count}),
    ], {"A": 44, "B": 38, "C": 38, **{letter: 16 for letter in "DEFGHIJKLMNOPQRSTUVWXYZ"}})
    notes = pd.DataFrame({"notes": [
        f"Rows used: {len(df):,} events from {df['ticker'].nunique():,} companies, the table of Task 2 (processed/classification_events.csv), unchanged: already filtered for price_data_unreliable and blank features.",
        f"Features: the 8 of Task 2 ({', '.join(FEATURES)}), encoded by the same encode() in data_phase2/classification_events.py: {encoded.shape[1]} encoded columns. The reaction label and the CAR are not inputs.",
        f"Preprocessing difference from Task 2: {scaler_note}",
        "Why these k values:",
        *reasons,
        f"k-means: scikit-learn KMeans, random_state={SEED}, n_init=10, otherwise default settings. Nothing was tuned. Clusters are renumbered 1..k from the largest down.",
        f"Silhouette: computed on a fixed random sample of {SILHOUETTE_SAMPLE:,} rows (random_state={SEED}), not on all rows.",
        sse_read, knee_read, silhouette_read,
        "Alignment: purity is the share of rows in their cluster's most common label. NMI runs from 0 (no shared information) to 1. ARI is 0 for chance agreement and 1 for identical groupings. k=4 has no label and is not checked.",
        "Values are outputs of data_phase2/clustering_events.py, stored as values rather than formulas. Charts: charts2/output/shared/29_clustering_elbow.png, 30_clustering_silhouette.png and 31_clustering_k5_profile.png.",
        "T3_Clustering_Data holds event_id, the four cluster columns and reaction_category. processed/clustering_events.csv holds the same rows plus the direction and reaction_3way reference labels.",
        f"The k=5 clusters in plain terms (size_category is quartiles of this dataset's companies, so Mega-cap starts at ${mega_floor / 1e9:,.1f}B here):",
        *[f"Cluster {row.cluster}: {row.description}. Mean market_cap ${row.mean_market_cap / 1e9:,.2f}B. Strong share {row.strong_share:.1%}, Up share {row.up_share:.1%}." for row in profiles.itertuples()],
        "Comparison and verdict:",
        *comparison,
        "Issues found:",
        *[f"{i}. {flag}" for i, flag in enumerate(flags, start=1)],
    ]})
    ws = write_sheet(wb, "T3_Clustering_Notes", notes, widths={"notes": 120})
    for cells in ws.iter_rows(min_row=2):
        cells[0].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(WORKBOOK)

    say("\nWROTE")
    say(f"  processed/clustering_events.csv ({len(assigned):,} rows), clustering_metrics.csv, clustering_k5_profile.csv")
    say(f"  processed/{WORKBOOK.name}: T3_Clustering_Data, T3_Clustering_Results, T3_Clustering_Notes")
    (PROC / "clustering_events_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
