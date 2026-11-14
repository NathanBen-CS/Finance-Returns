"""
EECS 4412 Phase 2, Task 2 -- classify each event's reaction_category with a decision tree and Naive Bayes.

  1. take the event rows of processed/full_dataset.csv and drop price_data_unreliable rows, then
     rows with a blank feature (nothing is imputed)
  2. encode: size_category and filing_timeliness as ordered numbers, log1p then z-score market_cap
     and employees, one-hot sector, exchange, event_category and filing day (z-score fitted on train only)
  3. stratified 80/20 split, random_state 42
  4. decision tree (max_depth None, 5, 10; depth kept by 5-fold accuracy on the training rows) and GaussianNB
  5. accuracy, per-class precision/recall/F1, confusion matrices, one-vs-rest ROC curves and macro AUC
  6. sensitivity checks on anything that could flatter the scores (reported only, not in the main results)

Run with the project venv (needs sklearn + openpyxl):  ..\\..\\.venv\\Scripts\\python.exe classification_events.py
Writes, in processed/:  classification_events.csv, classification_metrics.csv, classification_confusion.csv,
                        classification_roc_points.csv, classification_events_log.txt
                        220227286.xlsx (sheets T2_Classification_*; any other sheets are kept)
Writes, in raw/:        classification_dropped_companies.csv (companies whose events have a blank feature)
Charts come from ../charts2/classification_charts.py, which reads the files above.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pca_companies import BODY_FONT, HEADER_FILL, HEADER_FONT, SIZE_ORDER, log1p_clipped, write_sheet

HERE = Path(__file__).resolve().parent
PROC = HERE / "processed"
RAW = HERE / "raw"
WORKBOOK = PROC / "220227286.xlsx"

CLASSES = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]
TIMELINESS_ORDER = {"On-time": 1, "Slightly Late": 2, "Late": 3}   # ordinal in the Phase 1 spec
ORDINAL = {"size_category": SIZE_ORDER, "filing_timeliness": TIMELINESS_ORDER}
LOG_COLUMNS = ["market_cap", "employees"]
NOMINAL = ["sector", "exchange", "event_category", "filing_day_of_week"]
SOURCE_FEATURES = ["sector", "exchange", "size_category", "market_cap", "employees", "filing_timeliness", "event_category"]
FEATURES = [*SOURCE_FEATURES, "filing_day_of_week"]
ID_COLUMNS = ["event_id", "ticker", "trading_date", "filing_date"]
LABEL = "reaction_category"

# Every other column of full_dataset.csv, and why it is not a feature.
EXCLUDED = {
    "trailing_volatility_1y": "excluded on purpose: a per-company snapshot, and the standard deviation of returns is tied to CAR size by formula",
    "volatility_tier": "quartiles of trailing_volatility_1y, so the same problem",
    "car_minus1_plus1": "defines the label",
    "car_minus1_plus5": "contains the label's window",
    "abnormal_return": "price-derived, inside the label's window",
    "daily_return": "price-derived, inside the label's window",
    "market_return": "price-derived, inside the label's window",
    "volume_change_pct": "measured on day 0, so not known before the reaction",
    "price_data_unreliable": "used as the row filter; it is False on every kept row",
    "revenue_ttm": "not requested; a third size measure, also a snapshot, 290 companies blank",
    "sub_industry": "already rejected in Task 1: it nests inside sector and adds about 140 one-hot columns",
    "item_codes": "event_category is built from it; a list of codes, not one clean category",
    "num_items_disclosed": "not requested; counts the codes in item_codes",
    "report_date": "a date, not a feature; it anchors the label's window",
    "event_seq": "position in the company's history, not a concept requested",
    "event_id": "row label only", "ticker": "row label only", "trading_date": "row label only", "filing_date": "row label only",
    "accession_number": "identifier", "company_name": "identifier", "cik": "identifier", "sic_code": "identifier-like, overlaps sector",
    "sic_description": "identifier-like, overlaps sector", "selection_method": "zero variance",
}
DEPTHS = [None, 5, 10]
SEED = 42
ROC_POINTS = 800   # most points kept per stored curve; AUCs always use the full curve

log_lines = []


def say(text=""):
    """Print a line and keep it for classification_events_log.txt."""
    print(text, flush=True)
    log_lines.append(text)


def percent(n, total):
    return f"{n:,} ({n / total:.2%})"


def encode(df, train_rows, nominal=NOMINAL, ordinal=ORDINAL, log_columns=LOG_COLUMNS):
    """The features as numbers. The arguments are only changed by the sensitivity checks in main."""
    x = pd.DataFrame({col: df[col].map(order).astype(float) for col, order in ordinal.items()}, index=df.index)
    for col in log_columns:
        logged = log1p_clipped(df[col].astype(float)).to_frame()
        x[f"log_{col}"] = StandardScaler().fit(logged.iloc[train_rows]).transform(logged).ravel()
    one_hots = [pd.get_dummies(df[col], prefix=col, prefix_sep="=").astype(float) for col in nominal]
    return pd.concat([x, *one_hots], axis=1)


def tree_model(depth):
    return DecisionTreeClassifier(max_depth=depth, random_state=SEED)


def in_class_order(model, x):
    """predict_proba with its columns in CLASSES order rather than alphabetical."""
    order = [list(model.classes_).index(c) for c in CLASSES]
    return model.predict_proba(x)[:, order]


def predictions(model, x):
    return np.array(CLASSES)[in_class_order(model, x).argmax(axis=1)]


def macro_auc(y, scores):
    return np.mean([roc_auc_score(y == c, scores[:, i]) for i, c in enumerate(CLASSES)])


def evaluate(model, x, y, train_rows, test_rows):
    """Everything reported for one fitted model, on the test rows."""
    scores = in_class_order(model, x[test_rows])
    predicted = np.array(CLASSES)[scores.argmax(axis=1)]
    truth = y[test_rows]
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=CLASSES, zero_division=0)
    return {
        "train_accuracy": accuracy_score(y[train_rows], predictions(model, x[train_rows])),
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, predicted, labels=CLASSES, average="weighted", zero_division=0),
        "macro_auc": macro_auc(truth, scores),
        "per_class": pd.DataFrame({"class": CLASSES, "precision": precision, "recall": recall, "f1": f1, "support": support,
                                   "predicted_share": [(predicted == c).mean() for c in CLASSES],
                                   "auc": [roc_auc_score(truth == c, scores[:, i]) for i, c in enumerate(CLASSES)]}),
        "confusion": confusion_matrix(truth, predicted, labels=CLASSES),
        "scores": scores,
        "predicted": predicted,
    }


def roc_points(model_name, truth, scores):
    """One-vs-rest curves per class plus their macro average on a common grid."""
    grid = np.linspace(0, 1, 201)
    rows, curves = [], []
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(truth == c, scores[:, i])
        curves.append(np.interp(grid, fpr, tpr))
        keep = np.unique(np.linspace(0, len(fpr) - 1, ROC_POINTS).astype(int))
        rows.append(pd.DataFrame({"model": model_name, "curve": c, "fpr": fpr[keep], "tpr": tpr[keep]}))
    rows.append(pd.DataFrame({"model": model_name, "curve": "Macro average", "fpr": grid, "tpr": np.mean(curves, axis=0)}))
    return pd.concat(rows)


def sign_accuracy(truth, predicted):
    """Among rows where neither the truth nor the prediction is Neutral: did the prediction get the side right?"""
    side = {"Strong Negative": -1, "Negative": -1, "Neutral": 0, "Positive": 1, "Strong Positive": 1}
    t, p = pd.Series(truth).map(side).to_numpy(), pd.Series(predicted).map(side).to_numpy()
    both = (t != 0) & (p != 0)
    return (t[both] == p[both]).mean()


def check_split(x, y, train_rows, test_rows, depth):
    """Fit the kept tree and Naive Bayes on one alternative split or feature set."""
    tree = tree_model(depth).fit(x[train_rows], y[train_rows])
    bayes = GaussianNB().fit(x[train_rows], y[train_rows])
    return {"tree_accuracy": accuracy_score(y[test_rows], predictions(tree, x[test_rows])),
            "tree_macro_auc": macro_auc(y[test_rows], in_class_order(tree, x[test_rows])),
            "nb_accuracy": accuracy_score(y[test_rows], predictions(bayes, x[test_rows])),
            "nb_macro_auc": macro_auc(y[test_rows], in_class_order(bayes, x[test_rows])),
            "majority_baseline": pd.Series(y[test_rows]).value_counts(normalize=True).max()}


def write_blocks(wb, name, blocks, widths):
    """One sheet of stacked tables, each under a bold title, in the workbook's header and body style."""
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    row = 1
    for title, table, formats in blocks:
        ws.cell(row, 1, title).font = Font(name="Arial", bold=True)
        row += 1
        for col_idx, col in enumerate(table.columns, start=1):
            cell = ws.cell(row, col_idx, col)
            cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        for values in table.itertuples(index=False):
            row += 1
            for col_idx, (col, value) in enumerate(zip(table.columns, values), start=1):
                value = value.item() if isinstance(value, np.generic) else value
                cell = ws.cell(row, col_idx, None if value != value else value)   # NaN stays a blank cell
                cell.font = BODY_FONT
                if col in formats:
                    cell.number_format = formats[col]
        row += 2
    for letter, width in widths.items():
        ws.column_dimensions[letter].width = width
    return ws


def main():
    say("EECS 4412 Phase 2, Task 2: classification of reaction_category")
    say("=" * 70)
    read = dict(keep_default_na=False, na_values=[""])   # real tickers such as "NA" stay strings
    full = pd.read_csv(PROC / "full_dataset.csv", parse_dates=["filing_date", "report_date", "trading_date"], **read)
    assert set(EXCLUDED) | set(SOURCE_FEATURES) | {LABEL} == set(full.columns), "every column needs a decision"
    assert (full["selection_method"].nunique() == 1) and not set(EXCLUDED) & set(FEATURES)
    say(f"Loaded processed/full_dataset.csv: {len(full):,} events, {full['ticker'].nunique():,} companies, event-level rows.")

    # ---------------------------------------------------------------- 1. filter
    say("\nSTEP 1: ROWS")
    reliable = full[~full["price_data_unreliable"]]
    say(f"  price_data_unreliable == True: {len(full) - len(reliable):,} rows dropped, {len(reliable):,} left.")
    say("  Blanks in the features, on the remaining rows:")
    for col in SOURCE_FEATURES:
        say(f"    {col:<20}{reliable[col].isna().sum():>8,} ({reliable[col].isna().mean():.2%})")
    blank = reliable[SOURCE_FEATURES].isna().any(axis=1)
    say(f"  Rows with a blank feature: {blank.sum():,} from {reliable.loc[blank, 'ticker'].nunique()} companies. Dropped, as in Task 1; nothing is imputed.")
    say("  Dropped share by size: " + ", ".join(f"{s} {blank[reliable['size_category'] == s].mean():.1%}" for s in SIZE_ORDER))
    say("  Dropped share by class: " + ", ".join(f"{c} {blank[reliable[LABEL] == c].mean():.1%}" for c in CLASSES))
    dropped = reliable[blank].groupby("ticker").agg(company_name=("company_name", "first"), size_category=("size_category", "first"),
                                                    events_dropped=("event_id", "size")).reset_index()
    dropped["missing_columns"] = [",".join(c for c in SOURCE_FEATURES if reliable.loc[reliable["ticker"] == t, c].isna().any()) for t in dropped["ticker"]]
    dropped.to_csv(RAW / "classification_dropped_companies.csv", index=False)
    df = reliable[~blank].sort_values("event_id").reset_index(drop=True)
    df["filing_day_of_week"] = df["filing_date"].dt.day_name()
    for col in LOG_COLUMNS:
        df[col] = df[col].astype("int64")
    assert df[FEATURES].notna().all().all() and df["event_id"].is_unique
    say(f"  ROWS USED: {len(df):,} events from {df['ticker'].nunique():,} companies ({len(df) / len(full):.2%} of all events).")

    # ---------------------------------------------------------------- 2. label
    say("\nSTEP 2: LABEL")
    say("  reaction_category, unchanged from Phase 1: CAR[-1,+1] < -3% Strong Negative; < -0.5% Negative;")
    say("  <= +0.5% Neutral; <= +3% Positive; otherwise Strong Positive.")
    car = df["car_minus1_plus1"].to_numpy()
    thresholds = np.select([car < -0.03, car < -0.005, car <= 0.005, car <= 0.03], CLASSES[:4], CLASSES[4])
    mismatch = thresholds != df[LABEL].to_numpy()
    at_boundary = np.isclose(np.abs(car[mismatch]), 0.03, atol=1e-6) | np.isclose(np.abs(car[mismatch]), 0.005, atol=1e-6)
    assert at_boundary.all()
    say(f"  Check against the stored CAR: {mismatch.sum()} rows differ, all exactly at a boundary after rounding to 6 decimals. Labels are kept as stored.")
    counts = df[LABEL].value_counts().reindex(CLASSES)
    say("  Class distribution:")
    for c in CLASSES:
        say(f"    {c:<18}{percent(counts[c], len(df)):>20}")

    # ---------------------------------------------------------------- 3. features
    say("\nSTEP 3: FEATURES")
    say("  sector (nominal, one-hot): industries may differ in news sensitivity")
    say("  exchange (nominal, one-hot): rough proxy for visibility and quality tier")
    say("  size_category (ordinal: Small=1, Mid=2, Large=3, Mega-cap=4, as in Phase 1 and Task 1)")
    say("  market_cap (ratio, log1p then z-score): continuous size signal")
    say("  employees (ratio, log1p then z-score): size and maturity proxy")
    say("  filing_timeliness (ordinal in the Phase 1 spec: On-time=1, Slightly Late=2, Late=3): compliance signal")
    say("  event_category (nominal, one-hot): the kind of news, built from the 8-K item codes")
    say("  filing_day_of_week (nominal, one-hot): day name of filing_date; the only new column")
    for col in LOG_COLUMNS:
        say(f"  Skew of {col}: {df[col].skew():.2f} raw, {log1p_clipped(df[col].astype(float)).skew():.2f} after log1p, so log1p is used. No value is <= 0.")
    say("\n  Columns not used:")
    for col, why in EXCLUDED.items():
        say(f"    {col:<24}{why}")
    say("  Nothing was excluded for being more than 50% blank (the highest is employees, "
        f"{reliable['employees'].isna().mean():.2%}). selection_method is the only zero-variance column.")

    # ---------------------------------------------------------------- 4. split and encode
    y = df[LABEL].to_numpy()
    train_rows, test_rows = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    is_test = np.zeros(len(df), bool)
    is_test[test_rows] = True
    df["split"] = np.where(is_test, "test", "train")
    encoded = encode(df, train_rows)
    x = encoded.to_numpy()
    say("\nSTEP 4: SPLIT AND ENCODING")
    say(f"  Stratified on reaction_category, 80/20, random_state={SEED}: {len(train_rows):,} train, {len(test_rows):,} test.")
    split_sizes = pd.DataFrame({"class": CLASSES, "total": counts.to_numpy(), "share": counts.to_numpy() / len(df),
                                "train": [(y[train_rows] == c).sum() for c in CLASSES], "test": [(y[test_rows] == c).sum() for c in CLASSES]})
    for row in split_sizes.itertuples():
        say(f"    {row[1]:<18} train {row.train:>7,}  test {row.test:>7,}")
    say(f"  {len(FEATURES)} features become {x.shape[1]} encoded columns. The z-score of the two log columns is fitted on the training rows only.")
    say("  size_category and filing_timeliness stay as ordered numbers and the one-hot columns stay 0/1.")
    say("  A tree ignores feature scale, and GaussianNB fits a separate spread per column, so standardising barely changes either model.")

    # ---------------------------------------------------------------- 5. tree depth
    baseline = counts.max() / len(df)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    say("\nSTEP 5: DECISION TREE DEPTH")
    say(f"  Always guessing the largest class ({counts.idxmax()}) scores {baseline:.2%}.")
    say("  max_depth  leaves  train acc  test acc    gap  5-fold acc  test macro AUC")
    depth_rows, fitted = [], {}
    for depth in DEPTHS:
        model = tree_model(depth).fit(x[train_rows], y[train_rows])
        result = evaluate(model, x, y, train_rows, test_rows)
        cv = cross_val_score(tree_model(depth), x[train_rows], y[train_rows], cv=folds).mean()
        fitted[depth] = (model, result)
        depth_rows.append({"max_depth": "None" if depth is None else depth, "leaves": model.get_n_leaves(), "actual_depth": model.get_depth(),
                           "train_accuracy": result["train_accuracy"], "test_accuracy": result["accuracy"],
                           "gap": result["train_accuracy"] - result["accuracy"], "cv_accuracy_on_train": cv, "test_macro_auc": result["macro_auc"]})
        say(f"  {str(depth):<9}{model.get_n_leaves():>8,}{result['train_accuracy']:>11.2%}{result['accuracy']:>10.2%}"
            f"{result['train_accuracy'] - result['accuracy']:>7.2%}{cv:>12.2%}{result['macro_auc']:>16.4f}")
    depth_table = pd.DataFrame(depth_rows)
    kept_position = depth_table["cv_accuracy_on_train"].idxmax()
    kept_depth = DEPTHS[kept_position]
    tree, tree_result = fitted[kept_depth]
    say(f"  Kept max_depth={kept_depth}: the best 5-fold accuracy on the training rows, so the test rows played no part in the choice.")
    bayes = GaussianNB().fit(x[train_rows], y[train_rows])
    bayes_result = evaluate(bayes, x, y, train_rows, test_rows)
    models = {f"Decision Tree (max_depth={kept_depth})": tree_result, "Gaussian Naive Bayes": bayes_result}

    # ---------------------------------------------------------------- 6. results
    say("\nSTEP 6: RESULTS ON THE TEST ROWS")
    per_class_rows, confusion_rows, curves = [], [], []
    for name, result in models.items():
        say(f"\n  {name}")
        say(f"    accuracy {result['accuracy']:.2%} (train {result['train_accuracy']:.2%}, largest-class baseline {baseline:.2%}) | "
            f"macro F1 {result['macro_f1']:.4f} | weighted F1 {result['weighted_f1']:.4f} | macro AUC {result['macro_auc']:.4f}")
        say(f"    {'class':<18}{'precision':>10}{'recall':>9}{'F1':>8}{'AUC':>8}{'support':>9}{'share predicted':>17}")
        for row in result["per_class"].itertuples():
            say(f"    {row[1]:<18}{row.precision:>10.3f}{row.recall:>9.3f}{row.f1:>8.3f}{row.auc:>8.3f}{row.support:>9,}{row.predicted_share:>17.1%}")
        per_class = result["per_class"].copy()
        per_class.insert(0, "model", name)
        per_class_rows.append(per_class)
        matrix = pd.DataFrame(result["confusion"], columns=[f"predicted {c}" for c in CLASSES])
        matrix.insert(0, "true class", CLASSES)
        matrix.insert(0, "model", name)
        confusion_rows.append(matrix)
        say("    confusion matrix (rows true, columns predicted, in the class order above):")
        for c, values in zip(CLASSES, result["confusion"]):
            say(f"      {c:<18}" + "".join(f"{v:>8,}" for v in values))
        curves.append(roc_points(name, y[test_rows], result["scores"]))
    per_class = pd.concat(per_class_rows, ignore_index=True)
    confusion = pd.concat(confusion_rows, ignore_index=True)
    overall = pd.DataFrame([{"model": name, "train_accuracy": r["train_accuracy"], "test_accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                             "weighted_f1": r["weighted_f1"], "macro_auc": r["macro_auc"], "largest_class_baseline": baseline} for name, r in models.items()])

    # ---------------------------------------------------------------- 7. comparison
    tree_name, bayes_name = models
    say("\nSTEP 7: COMPARISON")
    winners = []
    for metric in ("f1", "auc"):
        tree_values, bayes_values = tree_result["per_class"][metric], bayes_result["per_class"][metric]
        tree_wins = [c for c, a, b in zip(CLASSES, tree_values, bayes_values) if a > b]
        bayes_wins = [c for c, a, b in zip(CLASSES, tree_values, bayes_values) if b > a]
        winners.append(f"{metric.upper()}: tree wins {', '.join(tree_wins) or 'none'}; Naive Bayes wins {', '.join(bayes_wins) or 'none'}.")
        say(f"  {winners[-1]}")
    sides = {name: sign_accuracy(y[test_rows], r["predicted"]) for name, r in models.items()}
    true_up = np.isin(y[test_rows], CLASSES[3:]).sum() / (~np.isin(y[test_rows], ["Neutral"])).sum()
    say(f"  Side of zero, among rows where neither truth nor prediction is Neutral: tree {sides[tree_name]:.2%}, Naive Bayes {sides[bayes_name]:.2%}; "
        f"{true_up:.2%} of these true classes are up.")
    importance = pd.Series(tree.feature_importances_, index=encoded.columns).sort_values(ascending=False)
    say("  Largest tree importances: " + ", ".join(f"{name} {value:.3f}" for name, value in importance.head(6).items()))
    is_earnings = df["event_category"] == "Financial/Earnings"
    is_strong = df[LABEL].str.startswith("Strong")
    strong_rate = {size: (is_strong[(df["size_category"] == size) & is_earnings].mean(), is_strong[(df["size_category"] == size) & ~is_earnings].mean()) for size in SIZE_ORDER}
    sector_strong = is_strong.groupby(df["sector"]).mean().sort_values()
    size_strong = is_strong.groupby(df["size_category"]).mean()
    tree_class, bayes_class = tree_result["per_class"].set_index("class"), bayes_result["per_class"].set_index("class")
    comparison = [
        f"The tree wins overall: accuracy {tree_result['accuracy']:.2%} vs {bayes_result['accuracy']:.2%}, macro F1 {tree_result['macro_f1']:.3f} vs {bayes_result['macro_f1']:.3f}, "
        f"macro AUC {tree_result['macro_auc']:.3f} vs {bayes_result['macro_auc']:.3f}. Both sit only a little above the {baseline:.2%} largest-class baseline and the 0.5 AUC of guessing.",
        *winners,
        f"Naive Bayes wins F1 on Neutral and Positive because it predicts them far more often ({bayes_class.loc['Neutral', 'predicted_share']:.1%} and {bayes_class.loc['Positive', 'predicted_share']:.1%} of test rows, "
        f"against {tree_class.loc['Neutral', 'predicted_share']:.1%} and {tree_class.loc['Positive', 'predicted_share']:.1%} for the tree), not because it ranks them better: the tree has the higher AUC on both. "
        f"In return it almost never predicts Negative ({bayes_class.loc['Negative', 'predicted_share']:.1%} of rows, recall {bayes_class.loc['Negative', 'recall']:.3f}).",
        f"Why the tree does better: the effects combine. An earnings filing lifts the chance of a Strong reaction from {strong_rate['Mega-cap'][1]:.1%} to {strong_rate['Mega-cap'][0]:.1%} at Mega-cap companies "
        f"but only from {strong_rate['Small'][1]:.1%} to {strong_rate['Small'][0]:.1%} at Small ones. A tree can split on that combination. Naive Bayes multiplies one independent effect per column, so it cannot. "
        "This reading is supported by what did not help Naive Bayes when tested (see the sensitivity table): giving it one size column instead of three, or discrete inputs instead of bell curves, barely changes its AUC.",
        f"What both models learn is how large a reaction is, not which way it goes. The share of Strong reactions runs from {size_strong['Mega-cap']:.1%} at Mega-cap to {size_strong['Small']:.1%} at Small, and from "
        f"{sector_strong.iloc[0]:.1%} in {sector_strong.index[0]} to {sector_strong.iloc[-1]:.1%} in {sector_strong.index[-1]}. The side of zero is close to a coin flip: "
        f"when both truth and prediction have a side, the tree is right {sides[tree_name]:.1%} of the time and Naive Bayes {sides[bayes_name]:.1%}, and {1 - true_up:.1%} of true non-Neutral rows are down. "
        "This is why each model's most common mistake on the two Strong classes is the opposite Strong class.",
    ]
    say("\n  " + "\n  ".join(comparison))

    # ---------------------------------------------------------------- 8. things that look off, measured
    say("\nSTEP 8: CHECKS ON WHAT COULD FLATTER THESE SCORES (reported only; the results above use the requested split and features)")
    company_day = df["ticker"] + "|" + df["trading_date"].astype(str)
    train_key_set = set(company_day.iloc[train_rows])
    test_shares_day = company_day.iloc[test_rows].isin(train_key_set).mean()
    shared_day = company_day.duplicated(keep=False).sum()
    by_company_train, by_company_test = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(x, y, df["ticker"]))
    by_day_train, by_day_test = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(x, y, company_day))
    cutoff = df["trading_date"].quantile(0.8)
    early, late = np.flatnonzero(df["trading_date"] < cutoff), np.flatnonzero(df["trading_date"] >= cutoff)
    suspects = pd.read_csv(RAW / "price_one_way_suspects.csv", **read)["event_ids_in_window"].dropna()
    suspect_ids = {e for ids in suspects for e in ids.split(";")}
    is_suspect = df["event_id"].isin(suspect_ids).to_numpy()
    without_suspects = np.intersect1d(train_rows, np.flatnonzero(~is_suspect)), np.intersect1d(test_rows, np.flatnonzero(~is_suspect))

    def without(*groups):
        """The encoding with some features removed."""
        return encode(df, train_rows, nominal=[c for c in NOMINAL if c not in groups],
                      ordinal={k: v for k, v in ORDINAL.items() if k not in groups},
                      log_columns=[c for c in LOG_COLUMNS if c not in groups]).to_numpy()

    rare = ["exchange=PNK", "event_category=Security Incident"]
    checks = {
        "Results above (stratified 80/20)": (x, train_rows, test_rows),
        "No company in both train and test": (x, by_company_train, by_company_test),
        "No company-day in both train and test": (x, by_day_train, by_day_test),
        f"Train on the earliest 80%, test from {cutoff.date()}": (x, early, late),
        "Without filing_timeliness and filing_day_of_week": (without("filing_timeliness", "filing_day_of_week"), train_rows, test_rows),
        "Without size_category, market_cap, employees": (without("size_category", "market_cap", "employees"), train_rows, test_rows),
        "Without event_category": (without("event_category"), train_rows, test_rows),
        "Without the one-way price suspects": (x, *without_suspects),
        f"Without the rare columns {' and '.join(rare)}": (encoded.drop(columns=rare).to_numpy(), train_rows, test_rows),
        "market_cap as the only size column": (without("size_category", "employees"), train_rows, test_rows),
    }
    say(f"  {'check':<62}{'tree acc':>9}{'tree AUC':>9}{'NB acc':>8}{'NB AUC':>8}{'baseline':>9}")
    sensitivity_rows = []
    for name, (features, train, test) in checks.items():
        scores = check_split(features, y, train, test, kept_depth)
        sensitivity_rows.append({"check": name, **scores})
        say(f"  {name:<62}{scores['tree_accuracy']:>9.2%}{scores['tree_macro_auc']:>9.4f}{scores['nb_accuracy']:>8.2%}{scores['nb_macro_auc']:>8.4f}{scores['majority_baseline']:>9.2%}")
    seed_scores, seed_rare_nb = [], []
    no_rare_columns = encoded.drop(columns=rare).to_numpy()
    for seed in range(5):
        seed_train, seed_test = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
        seed_scores.append(check_split(x, y, seed_train, seed_test, kept_depth))
        seed_rare_nb.append(check_split(no_rare_columns, y, seed_train, seed_test, kept_depth)["nb_accuracy"])
    seed_scores = pd.DataFrame(seed_scores)
    say("  Five other random splits (random_state 0-4), lowest to highest: " + ", ".join(
        f"{label} {seed_scores[col].min():.4f}-{seed_scores[col].max():.4f}" for label, col in
        [("tree accuracy", "tree_accuracy"), ("tree AUC", "tree_macro_auc"), ("Naive Bayes accuracy", "nb_accuracy"), ("Naive Bayes AUC", "nb_macro_auc")]))
    say(f"  Naive Bayes accuracy on those splits without the rare columns: {min(seed_rare_nb):.4f}-{max(seed_rare_nb):.4f}")
    # the same features as categories, so Naive Bayes needs no bell curves; market_cap and employees become deciles
    discrete = pd.DataFrame({col: pd.factorize(df[col])[0] for col in (*NOMINAL, *ORDINAL)})
    for col in LOG_COLUMNS:
        discrete[col] = pd.qcut(df[col], 10, labels=False, duplicates="drop")
    categorical = CategoricalNB(min_categories=discrete.max().to_numpy() + 1).fit(discrete.iloc[train_rows], y[train_rows])
    categorical_auc = macro_auc(y[test_rows], in_class_order(categorical, discrete.iloc[test_rows]))
    categorical_accuracy = accuracy_score(y[test_rows], predictions(categorical, discrete.iloc[test_rows]))
    say(f"  CategoricalNB on the same features, with market_cap and employees in deciles: accuracy {categorical_accuracy:.2%}, macro AUC {categorical_auc:.4f}")
    sensitivity_rows.append({"check": "CategoricalNB instead of GaussianNB (Naive Bayes columns only)", "nb_accuracy": categorical_accuracy, "nb_macro_auc": categorical_auc,
                             "majority_baseline": baseline})
    sensitivity = pd.DataFrame(sensitivity_rows)
    by_company_deep = tree_model(None).fit(x[by_company_train], y[by_company_train])
    deep_unseen = accuracy_score(y[by_company_test], predictions(by_company_deep, x[by_company_test]))
    say(f"  max_depth=None on unseen companies: test accuracy {deep_unseen:.2%}, against {fitted[None][1]['accuracy']:.2%} on the requested split.")

    companies = df.drop_duplicates("ticker")
    lag = (df["filing_date"] - df["report_date"]).dt.days
    trading_days = pd.read_csv(HERE / "cache" / "prices" / "t_IDX_GSPC.csv.gz", index_col=0, parse_dates=True).index
    sessions_after = trading_days.searchsorted(df["filing_date"]) - trading_days.get_indexer(df["trading_date"])
    filed_after_window = sessions_after >= 2
    slightly_late = df[df["filing_timeliness"] == "Slightly Late"]
    within_sec_rule = (np.busday_count(slightly_late["report_date"].to_numpy().astype("datetime64[D]"),
                                       slightly_late["filing_date"].to_numpy().astype("datetime64[D]")) <= 4).mean()
    size_car_spearman = df["market_cap"].corr(df["car_minus1_plus1"].abs(), method="spearman")
    volatility_car_spearman = df["trailing_volatility_1y"].corr(df["car_minus1_plus1"].abs(), method="spearman")
    size_volatility = companies["market_cap"].corr(companies["trailing_volatility_1y"], method="spearman")
    same_day_unique = df.groupby(["ticker", "trading_date"])[LABEL].nunique().max() == 1
    assert same_day_unique
    row = {r["check"]: r for r in sensitivity_rows}
    no_company = row["No company in both train and test"]
    no_size = row["Without size_category, market_cap, employees"]
    no_timing = row["Without filing_timeliness and filing_day_of_week"]
    no_event = row["Without event_category"]
    no_day = row["No company-day in both train and test"]
    no_rare = row[f"Without the rare columns {' and '.join(rare)}"]
    no_suspects = row["Without the one-way price suspects"]
    latest = row[f"Train on the earliest 80%, test from {cutoff.date()}"]
    one_size = row["market_cap as the only size column"]
    neutral_share = tree_result["per_class"].set_index("class").loc["Neutral", "predicted_share"]
    flags = [
        f"Weak signal: the tree reaches {tree_result['accuracy']:.2%} and Naive Bayes {bayes_result['accuracy']:.2%}, against {baseline:.2%} for always guessing {counts.idxmax()}. "
        "The features describe the company and the kind of filing, not the news itself, so this is expected.",
        f"Snapshot limitation: size_category, market_cap and employees are not point-in-time. Each of the {df['ticker'].nunique():,} companies has one value for all its events, taken in October 2026 and "
        f"attached to events back to {df['trading_date'].min().date()}. sector and exchange are snapshots too. Today's market_cap also reflects returns earned after the earlier events, so this is look-ahead.",
        f"market_cap works as a company ID: {companies['market_cap'].nunique():,} distinct values for {len(companies):,} companies. An unlimited tree can use it to memorise a company's history. "
        f"Depth 10 shows little of this: with no company in both train and test, tree accuracy moves from {tree_result['accuracy']:.2%} to {no_company['tree_accuracy']:.2%} and macro AUC from {tree_result['macro_auc']:.4f} to {no_company['tree_macro_auc']:.4f}. "
        f"The unlimited tree falls to {deep_unseen:.2%} on unseen companies.",
        f"Size partly stands in for the excluded volatility: across companies, Spearman(market_cap, trailing_volatility_1y) is {size_volatility:+.2f}. Across events, Spearman with |CAR[-1,+1]| is {size_car_spearman:+.2f} for market_cap and {volatility_car_spearman:+.2f} for volatility. "
        f"No formula ties them together, but they are not independent. Removing all three size columns takes macro AUC from {tree_result['macro_auc']:.4f} to {no_size['tree_macro_auc']:.4f} for the tree and from {bayes_result['macro_auc']:.4f} to {no_size['nb_macro_auc']:.4f} for Naive Bayes.",
        f"size_category is pd.qcut of market_cap, so it holds no extra information, and Naive Bayes treats the three size columns as independent evidence. Both are kept as requested. "
        f"Using market_cap as the only size column barely changes the scores (Naive Bayes macro AUC {one_size['nb_macro_auc']:.4f}, tree {one_size['tree_macro_auc']:.4f}).",
        f"Timing: filing_timeliness and filing_day_of_week are known at filing, but the label's window is anchored on report_date, not filing_date. {percent(filed_after_window.sum(), len(df))} of events were filed after CAR[-1,+1] had closed, including every Slightly Late and Late row. "
        f"For those rows these two features are not known before the reaction. No formula links them to CAR, and removing both leaves the results almost unchanged (tree macro AUC {no_timing['tree_macro_auc']:.4f}, Naive Bayes {no_timing['nb_macro_auc']:.4f}).",
        f"filing_timeliness uses calendar days from Phase 1: {within_sec_rule:.1%} of Slightly Late filings are within 4 business days, so on time under the SEC's rule. {(lag < 0).sum()} row has a filing_date before report_date and the longest gap is {lag.max()} days; both are kept as filed.",
        f"Same-day filings: {percent(shared_day, len(df))} of rows share a company and trading day with another row, and always share its label because they share one CAR. A random split puts {test_shares_day:.2%} of test rows beside a training row with the same answer. "
        f"Keeping each company-day on one side gives tree accuracy {no_day['tree_accuracy']:.2%} and macro AUC {no_day['tree_macro_auc']:.4f}.",
        f"Naive Bayes is unstable on rare one-hot columns: {(df['event_category'] == 'Security Incident').sum()} rows are Security Incident and {(df['exchange'] == 'PNK').sum()} are PNK (one company). Their tiny variances differ by class and shift every row's score. "
        f"Over five other random splits its accuracy runs {seed_scores['nb_accuracy'].min():.2%} to {seed_scores['nb_accuracy'].max():.2%} while AUC stays at {seed_scores['nb_macro_auc'].min():.4f} to {seed_scores['nb_macro_auc'].max():.4f}. "
        f"On the company-day split it scores {no_day['nb_accuracy']:.2%}, below that split's {no_day['majority_baseline']:.2%} baseline. Without the two columns, accuracy is {no_rare['nb_accuracy']:.2%} on the main split and {min(seed_rare_nb):.2%} to {max(seed_rare_nb):.2%} on the five others. "
        f"The columns are kept as requested, so compare the models on AUC more than on Naive Bayes accuracy. The tree moves far less: {seed_scores['tree_accuracy'].min():.2%} to {seed_scores['tree_accuracy'].max():.2%}.",
        f"Neutral is the smallest class ({counts['Neutral'] / len(df):.2%}) and the tree predicts it for {neutral_share:.2%} of test rows, so its precision and recall there rest on few predictions.",
        f"Bad prices can remain: {is_suspect.sum()} kept events touch a one-way price suspect (raw/price_one_way_suspects.csv) and {(df['car_minus1_plus1'].abs() > 1).sum()} have |CAR[-1,+1]| above 100%, some of them real. The label only records which bucket a CAR falls in, and removing the suspects leaves tree accuracy at {no_suspects['tree_accuracy']:.2%}.",
        f"Blank employees is not random: dropped rows are {blank[reliable['size_category'] == 'Small'].mean():.1%} of Small but {blank[reliable['size_category'] == 'Mega-cap'].mean():.1%} of Mega-cap events, so the rows used lean toward larger companies.",
        f"Survivorship: the universe is today's SEC ticker list, so companies that disappeared since 2016 are absent, which is inherited from the Phase 2 dataset.",
        f"A random split also ignores time. Training on the earliest 80% and testing from {cutoff.date()} gives tree accuracy {latest['tree_accuracy']:.2%} and macro AUC {latest['tree_macro_auc']:.4f}, against a baseline of {latest['majority_baseline']:.2%} because the class mix shifts. "
        f"That is only {(latest['tree_accuracy'] - latest['majority_baseline']) * 100:.2f} percentage points above the baseline, and Naive Bayes falls below it at {latest['nb_accuracy']:.2%}, so the requested random split makes accuracy look better than it would be on future filings.",
    ]
    say("\nFLAGS")
    for i, flag in enumerate(flags, start=1):
        say(f"  {i:>2}. {flag}")

    # ---------------------------------------------------------------- write everything
    data = df[[*ID_COLUMNS, *FEATURES, LABEL, "split"]].copy()
    data["trading_date"] = data["trading_date"].dt.date
    data["filing_date"] = data["filing_date"].dt.date
    data.to_csv(PROC / "classification_events.csv", index=False)
    metrics = per_class.copy()
    overall_long = overall.melt(id_vars="model", var_name="metric")
    pd.concat([overall_long, metrics.melt(id_vars=["model", "class"], var_name="metric")], ignore_index=True)[
        ["model", "class", "metric", "value"]].to_csv(PROC / "classification_metrics.csv", index=False)
    confusion.to_csv(PROC / "classification_confusion.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(PROC / "classification_roc_points.csv", index=False, float_format="%.6f")

    wb = load_workbook(WORKBOOK) if WORKBOOK.exists() else Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    write_sheet(wb, "T2_Classification_Data", data, {"trading_date": "yyyy-mm-dd", "filing_date": "yyyy-mm-dd", "market_cap": "#,##0", "employees": "#,##0"},
                {"event_id": 12, "sector": 24, "event_category": 24, "filing_day_of_week": 20, "reaction_category": 19, "market_cap": 20})
    share, count = "0.00%", "#,##0"
    score = "0.0000"
    fmt = {c: share for c in ("train_accuracy", "test_accuracy", "largest_class_baseline", "gap", "cv_accuracy_on_train", "share", "predicted_share",
                              "tree_accuracy", "nb_accuracy", "majority_baseline")}
    fmt.update({c: score for c in ("macro_f1", "weighted_f1", "macro_auc", "test_macro_auc", "precision", "recall", "f1", "auc", "tree_macro_auc", "nb_macro_auc")})
    fmt.update({c: count for c in ("support", "total", "train", "test", "leaves", *[f"predicted {c}" for c in CLASSES])})
    write_blocks(wb, "T2_Classification_Results", [
        ("Overall, on the test rows", overall, fmt),
        ("Per class, on the test rows", per_class, fmt),
        ("Confusion matrices, on the test rows (rows true, columns predicted)", confusion, fmt),
        ("Decision tree depths tried", depth_table, fmt),
        ("Class distribution and split sizes", split_sizes, fmt),
        ("Sensitivity checks (kept tree and Naive Bayes; not the main results)", sensitivity, fmt),
    ], {"A": 44, "B": 20, **{letter: 18 for letter in "CDEFGHI"}})
    notes = pd.DataFrame({"notes": [
        f"Rows used: {len(df):,} events from {df['ticker'].nunique():,} companies, out of {len(full):,}. Source: processed/full_dataset.csv.",
        f"Exclusions: {len(full) - len(reliable):,} rows with price_data_unreliable == True; then {blank.sum():,} rows with a blank feature ({len(dropped)} companies, raw/classification_dropped_companies.csv). Nothing was imputed.",
        "Label: reaction_category, already in the data and unchanged from Phase 1. CAR[-1,+1] < -3% Strong Negative; < -0.5% Negative; <= +0.5% Neutral; <= +3% Positive; otherwise Strong Positive.",
        "Class distribution: " + "; ".join(f"{c} {percent(counts[c], len(df))}" for c in CLASSES) + ".",
        "Features used: " + ", ".join(FEATURES) + ". T2_Classification_Data holds them as stored; the encoding below is applied by data_phase2/classification_events.py.",
        "Encoding: size_category Small=1, Mid=2, Large=3, Mega-cap=4; filing_timeliness On-time=1, Slightly Late=2, Late=3 (ordinal in the Phase 1 spec); log1p then z-score of market_cap and employees, fitted on the training rows; one-hot sector, exchange, event_category, filing_day_of_week. "
        f"{x.shape[1]} encoded columns.",
        "filing_day_of_week is the day name of filing_date.",
        f"Volatility exclusion: trailing_volatility_1y is excluded on purpose. It is a per-company snapshot, not point-in-time per filing, and it correlates mechanically with CAR size, so it would leak signal about the label. volatility_tier is quartiles of it and is excluded too.",
        "Other columns not used: " + "; ".join(f"{col}: {why}" for col, why in EXCLUDED.items() if col not in ("trailing_volatility_1y", "volatility_tier")) + ".",
        f"Split: stratified on reaction_category, 80/20, random_state={SEED}: {len(train_rows):,} train, {len(test_rows):,} test. " + "; ".join(f"{r[1]} {r.train:,} train, {r.test:,} test" for r in split_sizes.itertuples()) + ". The split column in T2_Classification_Data marks each row.",
        f"Decision tree: max_depth {', '.join(str(d) for d in DEPTHS)} tried. Kept {kept_depth}, the best 5-fold accuracy on the training rows; train {tree_result['train_accuracy']:.2%}, test {tree_result['accuracy']:.2%}. max_depth=None overfits: train {fitted[None][1]['train_accuracy']:.2%}, test {fitted[None][1]['accuracy']:.2%}. Depth 5 is simpler and close behind: {fitted[5][1]['accuracy']:.2%} test, macro AUC {fitted[5][1]['macro_auc']:.4f}.",
        "Naive Bayes: scikit-learn GaussianNB with default settings, on the same encoded columns and split.",
        "ROC: one-vs-rest per class; the macro AUC is the mean of the five class AUCs. Charts: charts2/output/shared/27_classification_roc_by_model.png and 28_classification_roc_comparison.png.",
        "Values are outputs of data_phase2/classification_events.py, stored as values rather than formulas.",
        "Comparison:",
        *comparison,
        "Issues found:",
        *[f"{i}. {flag}" for i, flag in enumerate(flags, start=1)],
    ]})
    ws = write_sheet(wb, "T2_Classification_Notes", notes, widths={"notes": 120})
    for cells in ws.iter_rows(min_row=2):
        cells[0].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(WORKBOOK)

    say("\nWROTE")
    say(f"  processed/classification_events.csv ({len(data):,} rows), classification_metrics.csv, classification_confusion.csv, classification_roc_points.csv")
    say(f"  processed/{WORKBOOK.name}: T2_Classification_Data, T2_Classification_Results, T2_Classification_Notes")
    say(f"  raw/classification_dropped_companies.csv ({len(dropped)} rows)")
    (PROC / "classification_events_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
