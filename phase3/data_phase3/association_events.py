"""
EECS 4412 Phase 3, Task 1 -- association analysis on the 8-K filings.

  1. one transaction per filing, holding 5 items: event_category, filing_timeliness, size_category,
     volatility_tier and reaction_category, each written as "attribute=value" (23 distinct items)
  2. support of every 2-item and 3-item itemset whose items come from different attributes
     (two values of one attribute never occur in the same filing, so those itemsets have support 0)
  3. the 5 most frequent itemsets of length >= 2, every rule they give, the 2 highest- and 2 lowest-confidence rules
  4. lift, Jaccard, cosine, Piatetsky-Shapiro and all-confidence for the 2 highest-confidence rules
  5. Supplementary Rule A: filing_timeliness=Late => reaction_category=X inside one event_category
  6. Supplementary Rule B: size_category x reaction_category inside event_category=Financial/Earnings

Run with the project venv:  ..\\..\\.venv\\Scripts\\python.exe association_events.py
Writes, in processed/:  association_events.csv, association_itemsets.csv, association_rules.csv,
                        association_rule_a_late.csv, association_rule_b_size.csv, association_events_log.txt
                        220227286.xlsx (sheets T1_Association_*; any other sheets are kept)
Charts come from ../charts3/association_charts.py, which reads the files above.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from phase3_common import (GROUPS, PHASE2_PROC, PROC, REACTIONS, load_filings, open_workbook, percent, save_workbook, say_to, write_blocks, write_notes,
                           write_sheet)

TOP = 5
RULE_A_CATEGORY = "Financial/Earnings"   # the largest event_category; the others are computed too
RULE_A_TIMELINESS = "Late"
RULE_B_CATEGORY = "Financial/Earnings"
MATCH_ON = ["size_category", "volatility_tier"]   # the company profile the matched rate in Rule A holds fixed

log_lines = []
say = say_to(log_lines)


def item(col, value):
    return f"{col}={value}"


def name(items):
    return "{" + ", ".join(items) + "}"


def all_itemsets(df):
    """Support of every 2- and 3-itemset across different attributes, including those that never occur."""
    rows = []
    for length in (2, 3):
        for cols in combinations(GROUPS, length):
            counts = df.groupby(list(cols)).size()
            for values in product(*(GROUPS[c] for c in cols)):
                items = [item(c, v) for c, v in zip(cols, values)]
                rows.append({"length": length, "itemset": name(items), **{f"item_{i}": it for i, it in enumerate(items, start=1)},
                             "count": int(counts.get(values, 0)), "support": counts.get(values, 0) / len(df)})
    table = pd.DataFrame(rows).sort_values(["count", "length", "itemset"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def items_of(row):
    return [row[c] for c in ("item_1", "item_2", "item_3") if isinstance(row.get(c), str)]


def measures(f11, f1p, fp1, n):
    """The contingency-table counts of rule X => Y and five interestingness measures (Module 005)."""
    return {"f11": f11, "f10": f1p - f11, "f01": fp1 - f11, "f00": n - f1p - fp1 + f11, "f1+": f1p, "f+1": fp1, "N": n,
            "support": f11 / n, "confidence": f11 / f1p,
            "lift": n * f11 / (f1p * fp1), "jaccard": f11 / (f1p + fp1 - f11), "cosine": f11 / np.sqrt(f1p * fp1),
            "piatetsky_shapiro": f11 / n - f1p * fp1 / n ** 2, "all_confidence": min(f11 / f1p, f11 / fp1)}


def cramers_v(table):
    chi2, p, dof, _ = chi2_contingency(table)
    return chi2, p, dof, np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))


def rule_a(df, category, timeliness):
    """timeliness => each reaction inside one event_category, with the lift against that category's base rate.
    matched_rate is what the antecedent's rows would show if each had the reaction mix of its own size x volatility cell."""
    sub = df if category is None else df[df["event_category"] == category]
    late = sub["filing_timeliness"] == timeliness
    base = sub["reaction_category"].value_counts(normalize=True).reindex(REACTIONS)
    cell_rates = pd.crosstab([sub[c] for c in MATCH_ON], sub["reaction_category"], normalize="index").reindex(columns=REACTIONS, fill_value=0)
    late_cells = sub[late].groupby(MATCH_ON).size()
    matched = (cell_rates.loc[late_cells.index].mul(late_cells, axis=0).sum() / late.sum()) if late.any() else pd.Series(np.nan, index=REACTIONS)
    chi2, p, dof, v = cramers_v(pd.crosstab(late, sub["reaction_category"])) if late.any() else (np.nan,) * 4
    rows = []
    for reaction in REACTIONS:
        both = int((late & (sub["reaction_category"] == reaction)).sum())
        confidence = both / late.sum() if late.any() else np.nan
        rows.append({"event_category": category or "(all filings)", "rule": f"filing_timeliness={timeliness} => reaction_category={reaction}",
                     "reaction_category": reaction, "rows_in_category": len(sub), "antecedent_rows": int(late.sum()), "rule_rows": both,
                     "confidence": confidence, "base_rate_in_category": base[reaction], "lift": confidence / base[reaction],
                     "profile_matched_rate": matched[reaction], "lift_vs_profile_matched": confidence / matched[reaction] if matched[reaction] > 0 else np.nan,
                     "chi2": chi2, "chi2_dof": dof, "chi2_p_value": p, "cramers_v": v})
    return pd.DataFrame(rows)


def main():
    say("EECS 4412 Phase 3, Task 1: association analysis of the 8-K filings")
    say("=" * 70)
    df, merged, unreliable = load_filings()
    n = len(df)
    say(f"Merged events + market_reactions (inner) + companies (left): {merged:,} rows; dropped {unreliable} with price_data_unreliable; "
        f"{n:,} filings from {df['ticker'].nunique():,} companies (asserted).")

    # ---------------------------------------------------------------- 1. transactions
    say("\nSTEP 1: ONE TRANSACTION PER FILING, 5 ITEMS EACH")
    transactions = df[["event_id", "ticker", "filing_date"]].copy()
    for col in GROUPS:
        transactions[col] = col + "=" + df[col]
    one_hot = pd.DataFrame({item(c, v): (df[c] == v).to_numpy() for c in GROUPS for v in GROUPS[c]})
    assert one_hot.shape[1] == 23 and (one_hot.sum(axis=1) == len(GROUPS)).all()
    singles = pd.DataFrame({"attribute": [c for c in GROUPS for _ in GROUPS[c]], "item": one_hot.columns, "count": one_hot.sum().to_numpy()})
    singles["support"] = singles["count"] / n
    say(f"  {one_hot.shape[1]} distinct items in {len(GROUPS)} attributes: " + ", ".join(f"{c} ({len(v)})" for c, v in GROUPS.items()) + ". Every filing holds exactly one item per attribute.")
    for row in singles.itertuples():
        say(f"    {row.item:<42}{row.count:>9,}{row.support:>9.2%}")

    # ---------------------------------------------------------------- 2. itemsets
    say("\nSTEP 2: SUPPORT OF EVERY 2- AND 3-ITEMSET ACROSS DIFFERENT ATTRIBUTES")
    itemsets = all_itemsets(df)
    by_length = itemsets.groupby("length")["count"].agg(["size", lambda s: (s > 0).sum()])
    say(f"  2-itemsets: {by_length.iloc[0, 0]} ({by_length.iloc[0, 1]} occur at least once); 3-itemsets: {by_length.iloc[1, 0]} ({by_length.iloc[1, 1]} occur at least once).")
    say("  Every one is counted exactly, so the result is what Apriori would return at any minimum support; nothing is pruned.")
    top = itemsets.head(TOP).copy()
    next_count = itemsets["count"].iloc[TOP]
    tie_at_cut = top["count"].iloc[-1] == next_count
    for row in top.to_dict("records"):
        assert one_hot[items_of(row)].all(axis=1).sum() == row["count"]   # recount by a second route
    say(f"  Top {TOP} by support (recounted from the one-hot table: identical):")
    for row in top.itertuples():
        say(f"    {row.rank}. {row.itemset:<92}{row.count:>9,}{row.support:>9.2%}")
    say(f"  Next itemset, rank {TOP + 1}: {itemsets['itemset'].iloc[TOP]} {itemsets['count'].iloc[TOP]:,} ({itemsets['support'].iloc[TOP]:.2%}). "
        + ("TIE at the cut-off." if tie_at_cut else "No tie at the cut-off."))
    best_three = itemsets[itemsets["length"] == 3].iloc[0]
    say(f"  Most frequent 3-itemset: {best_three['itemset']} at rank {best_three['rank']}, {best_three['count']:,} ({best_three['support']:.2%}).")

    # ---------------------------------------------------------------- 3. rules
    say("\nSTEP 3: EVERY RULE FROM THE TOP 5 ITEMSETS")
    count_of = lambda items: int(one_hot[list(items)].all(axis=1).sum())
    rule_rows = []
    for row in top.to_dict("records"):
        items = items_of(row)
        for size in range(1, len(items)):
            for antecedent in combinations(items, size):
                consequent = [i for i in items if i not in antecedent]
                rule_rows.append({"from_itemset_rank": row["rank"], "itemset": row["itemset"], "antecedent": name(antecedent), "consequent": name(consequent),
                                  "rule": f"{name(antecedent)} => {name(consequent)}", **measures(row["count"], count_of(antecedent), count_of(consequent), n)})
    rules = pd.DataFrame(rule_rows).sort_values(["confidence", "rule"], ascending=[False, True], kind="stable").reset_index(drop=True)
    rules.insert(0, "confidence_rank", np.arange(1, len(rules) + 1))
    rules["selected_as"] = ""
    rules.loc[:1, "selected_as"] = "highest confidence"
    rules.loc[len(rules) - 2:, "selected_as"] = "lowest confidence"
    conf = rules["confidence"].to_numpy()
    ties = {"between 2nd and 3rd highest": np.isclose(conf[1], conf[2]), "between 2nd and 3rd lowest": np.isclose(conf[-2], conf[-3])}
    say(f"  {len(rules)} rules (each itemset of length L gives 2^L - 2). Sorted by confidence:")
    say(f"    {'#':>2} {'rule':<100}{'support':>9}{'confidence':>12}{'lift':>8}  selected")
    for row in rules.itertuples():
        say(f"    {row.confidence_rank:>2} {row.rule:<100}{row.support:>9.2%}{row.confidence:>12.2%}{row.lift:>8.4f}  {row.selected_as}")
    say("  Ties at the selection boundaries: " + ", ".join(f"{k}: {'YES' if v else 'no'}" for k, v in ties.items()))

    say("\nSTEP 4: INTERESTINGNESS MEASURES FOR THE 2 HIGHEST-CONFIDENCE RULES")
    measure_cols = ["f11", "f10", "f01", "f00", "f1+", "f+1", "N", "support", "confidence", "lift", "jaccard", "cosine", "piatetsky_shapiro", "all_confidence"]
    best = rules.head(2)
    for row in best.to_dict("records"):
        say(f"  {row['rule']}")
        say(f"    f11 {row['f11']:,}  f10 {row['f10']:,}  f01 {row['f01']:,}  f00 {row['f00']:,}  (f1+ {row['f1+']:,}, f+1 {row['f+1']:,}, N {row['N']:,})")
        say(f"    lift {row['lift']:.4f}  Jaccard {row['jaccard']:.4f}  cosine {row['cosine']:.4f}  Piatetsky-Shapiro {row['piatetsky_shapiro']:+.6f}  all-confidence {row['all_confidence']:.4f}")
        assert row["f11"] + row["f10"] + row["f01"] + row["f00"] == n

    # ---------------------------------------------------------------- 5. Rule A
    say(f"\nSTEP 5: SUPPLEMENTARY RULE A -- filing_timeliness={RULE_A_TIMELINESS} => reaction_category=X, inside one event_category")
    say(f"  profile_matched_rate: the reaction mix the {RULE_A_TIMELINESS} rows would show if each had the rate of its own {' x '.join(MATCH_ON)} cell inside the category. "
        "lift_vs_profile_matched above 1 means more than the company profile of late filers explains.")
    rule_a_all = pd.concat([rule_a(df, c, RULE_A_TIMELINESS) for c in [None, *GROUPS["event_category"]]], ignore_index=True)
    for category, part in rule_a_all.groupby("event_category", sort=False):
        first = part.iloc[0]
        say(f"  {category}: {first['rows_in_category']:,} filings, {first['antecedent_rows']:,} of them {RULE_A_TIMELINESS}. "
            f"Chi-square ({RULE_A_TIMELINESS} vs not, x 5 reactions) {first['chi2']:.1f}, dof {first['chi2_dof']:.0f}, p {first['chi2_p_value']:.3g}, Cramer's V {first['cramers_v']:.4f}")
        if first["antecedent_rows"] == 0:
            continue
        for row in part.itertuples():
            say(f"    => {row.reaction_category:<16} {row.rule_rows:>6,} rows  confidence {row.confidence:>7.2%}  base {row.base_rate_in_category:>7.2%}  lift {row.lift:.4f}  "
                f"matched rate {row.profile_matched_rate:>7.2%}  lift vs matched {row.lift_vs_profile_matched:.4f}")
    rule_a_main = rule_a_all[rule_a_all["event_category"] == RULE_A_CATEGORY].reset_index(drop=True)
    timing = df.assign(days=(df["trading_date"] - df["filing_date"]).dt.days).groupby("filing_timeliness")["days"].median()
    late_ec = df.loc[df["filing_timeliness"] == RULE_A_TIMELINESS, "event_category"].value_counts()

    # ---------------------------------------------------------------- 6. Rule B
    say(f"\nSTEP 6: SUPPLEMENTARY RULE B -- size_category x reaction_category inside event_category={RULE_B_CATEGORY}")
    sub = df[df["event_category"] == RULE_B_CATEGORY]
    counts = pd.crosstab(sub["size_category"], sub["reaction_category"]).reindex(index=GROUPS["size_category"], columns=REACTIONS)
    base_b = counts.sum() / counts.to_numpy().sum()
    confidence_b = counts.div(counts.sum(axis=1), axis=0)
    lift_b = confidence_b / base_b
    rule_b = pd.DataFrame([{"size_category": s, "reaction_category": r, "rule": f"size_category={s} => reaction_category={r}", "size_rows": int(counts.loc[s].sum()),
                            "rule_rows": int(counts.loc[s, r]), "confidence": confidence_b.loc[s, r], "base_rate_in_category": base_b[r], "lift": lift_b.loc[s, r]}
                           for s in GROUPS["size_category"] for r in REACTIONS])
    chi2_b, p_b, dof_b, v_b = cramers_v(counts)
    v_vol = cramers_v(pd.crosstab(sub["volatility_tier"], sub["reaction_category"]))[3]
    say(f"  {len(sub):,} filings. Chi-square {chi2_b:,.1f}, dof {dof_b}, p {p_b:.3g}, Cramer's V {v_b:.4f} (volatility_tier x reaction in the same rows: V {v_vol:.4f})")
    say("  Lift (row: size_category, column: reaction_category):")
    say(lift_b.to_string(float_format=lambda v: f"{v:.4f}"))
    say("  Rows behind each cell:")
    say(counts.to_string())
    strong = ["Strong Negative", "Strong Positive"]
    strong_share = counts[strong].sum(axis=1) / counts.sum(axis=1)

    # ---------------------------------------------------------------- 7. flags
    mega_floor = pd.read_csv(PHASE2_PROC / "companies.csv", keep_default_na=False, na_values=[""]).query("size_category == 'Mega-cap'")["market_cap"].min()
    same_day = df.groupby(["ticker", "trading_date"])["event_id"].transform("size") > 1
    top_lifts = rules["lift"]
    flags = [
        f"The top itemsets follow the most common single items, not strong associations. filing_timeliness=On-time is {singles.set_index('item').loc['filing_timeliness=On-time', 'support']:.2%} of filings and event_category=Financial/Earnings {singles.set_index('item').loc['event_category=Financial/Earnings', 'support']:.2%}, "
        f"so the itemsets that pair them with other common items lead. The lift of all {len(rules)} rules runs {top_lifts.min():.4f} to {top_lifts.max():.4f}, close to 1 (independence).",
        f"Reaction day 0 is the first trading day on or after report_date (the event date), not filing_date. Median trading_date minus filing_date: On-time {timing['On-time']:.0f} days, Slightly Late {timing['Slightly Late']:.0f}, Late {timing['Late']:.0f}. "
        f"So for a Late filing, reaction_category measures the market around the underlying event, about a month before the 8-K was filed, not a reaction to the late filing. Rule A tests whether late filers had different event-day reactions, not whether lateness moved the price.",
        f"Rule A rests on few rows: {RULE_A_TIMELINESS} is {late_ec.sum():,} of {n:,} filings ({late_ec.sum() / n:.2%}), and {rule_a_main['antecedent_rows'].iloc[0]:,} of them are {RULE_A_CATEGORY}; the smallest cell behind a {RULE_A_CATEGORY} lift is {rule_a_main['rule_rows'].min():,} rows. "
        f"{RULE_A_TIMELINESS} filings by event_category: " + ", ".join(f"{k} {v:,}" for k, v in late_ec.items()) + ".",
        "Rule A controls for event_category (by fixing it) and, in the matched columns, for size_category x volatility_tier. It does not control for the company itself; companies that file late repeatedly count once per filing.",
        "volatility_tier is circular as a predictor of reaction size: it is quartiles of trailing_volatility_1y, a single October 2026 value computed partly from the same returns the reactions are measured on. Itemsets and rules pairing volatility_tier with a Strong reaction partly restate that.",
        f"size_category, volatility_tier and sector are October 2026 snapshots attached to filings back to {df['filing_date'].min().date()}. size_category is quartiles of this dataset's companies (Mega-cap starts at ${mega_floor / 1e9:,.1f}B here).",
        f"Rows are not independent: {percent(int(same_day.sum()), n)} share a company and trading day with another filing and therefore share its CAR and reaction_category. Support counts each of them; chi-square p-values treat them as independent, so read them as optimistic.",
    ]
    say("\nFLAGS")
    for i, flag in enumerate(flags, start=1):
        say(f"  {i}. {flag}")

    # ---------------------------------------------------------------- write everything
    transactions.to_csv(PROC / "association_events.csv", index=False)
    itemsets.to_csv(PROC / "association_itemsets.csv", index=False)
    rules.to_csv(PROC / "association_rules.csv", index=False)
    rule_a_all.to_csv(PROC / "association_rule_a_late.csv", index=False)
    rule_b.to_csv(PROC / "association_rule_b_size.csv", index=False)

    share, count, score = "0.00%", "#,##0", "0.0000"
    fmt = {c: share for c in ("support", "confidence", "base_rate_in_category", "profile_matched_rate", *REACTIONS)}
    fmt |= {c: count for c in ("count", "f11", "f10", "f01", "f00", "f1+", "f+1", "N", "rows_in_category", "antecedent_rows", "rule_rows", "size_rows")}
    fmt |= {c: score for c in ("lift", "jaccard", "cosine", "all_confidence", "lift_vs_profile_matched", "cramers_v")}
    fmt |= {"piatetsky_shapiro": "0.000000", "chi2": "#,##0.0", "chi2_p_value": "0.00E+00"}
    lift_wide = lift_b.reset_index().rename(columns={"size_category": "size_category (lift)"})
    count_wide = counts.reset_index().rename(columns={"size_category": "size_category (rows)"})
    wb = open_workbook()
    write_sheet(wb, "T1_Association_Data", transactions, {"filing_date": "yyyy-mm-dd"},
                {"event_id": 12, "ticker": 9, "filing_date": 12, "event_category": 40, "filing_timeliness": 32, "size_category": 26, "volatility_tier": 28, "reaction_category": 36})
    write_blocks(wb, "T1_Association_Results", [
        ("The 23 items (single-item support, for reference)", singles, fmt),
        (f"Top {TOP} itemsets by support, length >= 2 (all {len(itemsets)} itemsets are in processed/association_itemsets.csv)", top[["rank", "length", "itemset", "count", "support"]], fmt),
        ("Itemsets ranked 6 to 20, for context", itemsets.iloc[TOP:20][["rank", "length", "itemset", "count", "support"]], fmt),
        (f"All {len(rules)} rules from the top {TOP} itemsets, by confidence (2 highest and 2 lowest marked)", rules[["confidence_rank", "rule", "support", "confidence", "lift", "selected_as"]], fmt),
        ("Interestingness measures for the 2 highest-confidence rules (X => Y; f1+ = count of X, f+1 = count of Y)", best[["rule", *measure_cols]], fmt),
        (f"Supplementary Rule A: filing_timeliness={RULE_A_TIMELINESS} => reaction_category, inside event_category={RULE_A_CATEGORY}", rule_a_main.drop(columns=["event_category"]), fmt),
        (f"Supplementary Rule A for every event_category (and all filings), same columns", rule_a_all, fmt),
        (f"Supplementary Rule B: lift of size_category => reaction_category inside event_category={RULE_B_CATEGORY} (base rates inside that category)", lift_wide, {r: score for r in REACTIONS}),
        ("Supplementary Rule B: rows behind each cell", count_wide, {r: count for r in REACTIONS}),
        ("Supplementary Rule B: every cell as a rule", rule_b, fmt),
    ], {"A": 26, "B": 96, "C": 16, "D": 16, **{letter: 15 for letter in "EFGHIJKLMNOPQRSTUVW"}})
    a = rule_a_main.set_index("reaction_category")
    notes = [
        f"Rows used: {n:,} filings from {df['ticker'].nunique():,} companies: processed events.csv + market_reactions.csv (inner on event_id) + companies.csv (left on ticker) from Phase 2, minus {unreliable} rows with price_data_unreliable.",
        "Table: one transaction per filing with 5 items, one per attribute: event_category (7 values), filing_timeliness (3), size_category (4), volatility_tier (4), reaction_category (5); 23 distinct items written as attribute=value. T1_Association_Data holds it, with event_id, ticker and filing_date to trace each row.",
        "Tool: Python (pandas). Support is counted exactly for every 2- and 3-itemset whose items come from different attributes; two values of one attribute never share a filing, so those itemsets are skipped (support 0). "
        f"That is {by_length.iloc[0, 0]} 2-itemsets and {by_length.iloc[1, 0]} 3-itemsets, few enough to count all of them, which gives what Apriori would return at any minimum support. The top 5 were recounted from a one-hot table as a check.",
        "Rules: every split of each top-5 itemset into a non-empty antecedent and consequent. confidence = count(X and Y) / count(X). Measures use f11 = count(X and Y), f10 = X without Y, f01 = Y without X, f00 = neither, N = all filings.",
        "lift = N*f11 / (f1+ * f+1); Jaccard = f11 / (f1+ + f+1 - f11); cosine = f11 / sqrt(f1+ * f+1); Piatetsky-Shapiro = f11/N - f1+*f+1/N^2; all-confidence = min(f11/f1+, f11/f+1).",
        "Top 5 itemsets: " + "; ".join(f"{r.itemset} {r.support:.2%}" for r in top.itertuples()) + f". Rank 6 is {itemsets['support'].iloc[TOP]:.2%}" + (" (a tie at the cut-off)." if tie_at_cut else "."),
        "Highest confidence: " + "; ".join(f"{r.rule} {r.confidence:.2%}" for r in rules.head(2).itertuples()) + ". Lowest: " + "; ".join(f"{r.rule} {r.confidence:.2%}" for r in rules.tail(2).itertuples()) + ".",
        f"Rule A ({RULE_A_CATEGORY}, {a['antecedent_rows'].iloc[0]:,} {RULE_A_TIMELINESS} filings): " + "; ".join(f"{r} lift {a.loc[r, 'lift']:.3f} ({a.loc[r, 'rule_rows']:,} rows)" for r in REACTIONS)
        + f". Chi-square p {a['chi2_p_value'].iloc[0]:.3g}, Cramer's V {a['cramers_v'].iloc[0]:.4f}. Against the size x volatility matched rate: " + "; ".join(f"{r} {a.loc[r, 'lift_vs_profile_matched']:.3f}" for r in REACTIONS) + ".",
        f"Rule B ({RULE_B_CATEGORY}, {len(sub):,} filings): Strong share by size_category " + ", ".join(f"{s} {v:.1%}" for s, v in strong_share.items()) + f"; Cramer's V {v_b:.4f} (volatility_tier: {v_vol:.4f}).",
        "Values are outputs of data_phase3/association_events.py, stored as values rather than formulas. Every itemset is in processed/association_itemsets.csv and every rule in processed/association_rules.csv.",
        "Issues found:",
        *[f"{i}. {flag}" for i, flag in enumerate(flags, start=1)],
    ]
    write_notes(wb, "T1_Association_Notes", notes)
    save_workbook(wb)
    say("\nWROTE")
    say(f"  processed/association_events.csv ({len(transactions):,} rows), association_itemsets.csv ({len(itemsets)} rows), association_rules.csv ({len(rules)} rules),")
    say("  association_rule_a_late.csv, association_rule_b_size.csv; 220227286.xlsx: T1_Association_Data, T1_Association_Results, T1_Association_Notes")
    (PROC / "association_events_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
