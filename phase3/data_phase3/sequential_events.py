"""
EECS 4412 Phase 3, Task 2 -- sequential analysis: does a filing's reaction predict the company's next filing's reaction?

  1. per company (ticker), the filings in order of filing_date (ties broken by event_id)
  2. each consecutive pair is one transition (prev reaction_category -> curr reaction_category);
     a company's first filing has no previous one and is dropped
  3. pattern support = count of the pair / all transitions; the 2 most frequent patterns, and for each the rule
     prev => curr with confidence = count(prev, curr) / count(prev) and lift = confidence / curr's base rate
  4. robustness A: the same, with transitions only inside one (ticker, event_category)
  5. robustness B: the trading-day gap between the pair, split at its median into short and long
  6. extra checks (not in the brief, reported separately): pairs whose CAR[-1,+1] windows do not overlap,
     and a within-company shuffle that keeps each company's reaction mix but destroys the order

Run with the project venv:  ..\\..\\.venv\\Scripts\\python.exe sequential_events.py
Writes, in processed/:  sequential_events.csv, sequential_patterns.csv, sequential_summary.csv, sequential_events_log.txt
                        220227286.xlsx (sheets T2_Sequential_*; any other sheets are kept)
Charts come from ../charts3/sequential_charts.py, which reads the files above.
"""
import numpy as np
import pandas as pd

from phase3_common import PROC, REACTIONS, load_filings, open_workbook, percent, save_workbook, say_to, write_blocks, write_notes, write_sheet

TOP = 2
MIN_GAP_NO_OVERLAP = 3   # trading days; CAR[-1,+1] windows of two filings share no day from this gap on (Tier 0's MIN_GAP_CAR1)
SHUFFLES = 20
SEED = 42

MAIN = "All consecutive filings"
SAME_CATEGORY = "Robustness A: same event_category only"
SHORT, LONG = "Robustness B: short gap", "Robustness B: long gap"
NO_OVERLAP = f"Extra: gap >= {MIN_GAP_NO_OVERLAP} trading days (no shared CAR day)"

log_lines = []
say = say_to(log_lines)


def transitions(df, keys):
    """Each filing with the previous filing in its group (df must already be in sequence order)."""
    groups = df.groupby(keys, sort=False)
    t = df.assign(prev_event_id=groups["event_id"].shift(1), prev_filing_date=groups["filing_date"].shift(1),
                  prev_event_category=groups["event_category"].shift(1), prev_reaction=groups["reaction_category"].shift(1),
                  prev_trading_day=groups["trading_day"].shift(1))
    t = t[t["prev_reaction"].notna()].copy()
    t["signed_gap"] = (t["trading_day"] - t["prev_trading_day"]).astype(int)
    t["gap_trading_days"] = t["signed_gap"].abs()
    return t


def patterns(t, variant):
    """All 25 prev -> curr patterns with support, and the rule prev => curr with confidence and lift."""
    counts = pd.crosstab(t["prev_reaction"], t["reaction_category"]).reindex(index=REACTIONS, columns=REACTIONS, fill_value=0)
    total = counts.to_numpy().sum()
    base = counts.sum() / total   # curr's share among these transitions
    rows = [{"variant": variant, "pattern": f"<{a}> -> <{b}>", "prev_reaction": a, "curr_reaction": b, "count": int(counts.loc[a, b]),
             "support": counts.loc[a, b] / total, "prev_count": int(counts.loc[a].sum()), "confidence": counts.loc[a, b] / counts.loc[a].sum(),
             "curr_base_rate": base[b], "lift": counts.loc[a, b] / counts.loc[a].sum() / base[b]} for a in REACTIONS for b in REACTIONS]
    table = pd.DataFrame(rows).sort_values(["count", "pattern"], ascending=[False, True], kind="stable").reset_index(drop=True)
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table


def summary(t, table, variant):
    diagonal = table[table["prev_reaction"] == table["curr_reaction"]]
    prev_share = t["prev_reaction"].value_counts(normalize=True).reindex(REACTIONS)
    curr_share = t["reaction_category"].value_counts(normalize=True).reindex(REACTIONS)
    top = table.head(TOP)
    return {"variant": variant, "transitions": len(t), "companies": t["ticker"].nunique(), "median_gap_trading_days": t["gap_trading_days"].median(),
            "same_reaction_share": diagonal["count"].sum() / len(t), "same_reaction_if_independent": (prev_share * curr_share).sum(),
            **{f"pattern_{i}": r.pattern for i, r in enumerate(top.itertuples(), start=1)},
            **{f"pattern_{i}_{m}": getattr(r, m) for i, r in enumerate(top.itertuples(), start=1) for m in ("support", "confidence", "lift")}}


def main():
    say("EECS 4412 Phase 3, Task 2: sequential analysis of reactions across a company's filings")
    say("=" * 70)
    df, merged, unreliable = load_filings()
    n = len(df)
    say(f"Merged events + market_reactions (inner) + companies (left): {merged:,} rows; dropped {unreliable} with price_data_unreliable; "
        f"{n:,} filings from {df['ticker'].nunique():,} companies (asserted).")

    # ---------------------------------------------------------------- 1. sequences
    say("\nSTEP 1: ONE SEQUENCE PER COMPANY, IN FILING ORDER")
    df = df.sort_values(["ticker", "filing_date", "event_id"], kind="stable").reset_index(drop=True)
    calendar = np.sort(df["trading_date"].unique())   # Tier 0's market calendar: every distinct trading date in the data
    df["trading_day"] = np.searchsorted(calendar, df["trading_date"].to_numpy())
    per_company = df.groupby("ticker").size()
    say(f"  Sorted by ticker, filing_date, event_id. Sequence length per company: median {per_company.median():.0f}, min {per_company.min()}, max {per_company.max():,}.")
    say(f"  Trading-day calendar: the {len(calendar):,} distinct trading dates in the data ({str(calendar[0])[:10]} to {str(calendar[-1])[:10]}), as in network/tier0_prepare_data.py. "
        "The gap is counted between the two filings' reaction day 0 (trading_date).")

    # ---------------------------------------------------------------- 2. main transitions
    say("\nSTEP 2: CONSECUTIVE TRANSITIONS, ALL FILINGS")
    main_t = transitions(df, ["ticker"])
    assert len(main_t) == n - df["ticker"].nunique()
    median_gap = main_t["gap_trading_days"].median()
    main_t["same_event_category"] = main_t["event_category"] == main_t["prev_event_category"]
    main_t["gap_group"] = np.where(main_t["gap_trading_days"] <= median_gap, "short", "long")
    say(f"  {len(main_t):,} transitions ({n:,} filings minus each company's first). Gap in trading days: median {median_gap:.0f}, "
        f"25% {main_t['gap_trading_days'].quantile(0.25):.0f}, 75% {main_t['gap_trading_days'].quantile(0.75):.0f}, max {main_t['gap_trading_days'].max():,}.")
    negative = (main_t["signed_gap"] < 0).sum()
    say(f"  {negative:,} transitions have the earlier-filed filing's reaction day after the later one's (report_date order differs from filing_date order); their gap is taken as the absolute value.")

    variants = {MAIN: main_t, SAME_CATEGORY: None,
                SHORT: main_t[main_t["gap_group"] == "short"], LONG: main_t[main_t["gap_group"] == "long"],
                NO_OVERLAP: main_t[main_t["gap_trading_days"] >= MIN_GAP_NO_OVERLAP]}
    variants[SAME_CATEGORY] = transitions(df, ["ticker", "event_category"])
    tables = {name: patterns(t, name) for name, t in variants.items()}
    main_table = tables[MAIN]
    say("  All 25 patterns, by support:")
    say(f"    {'#':>3} {'pattern':<42}{'count':>9}{'support':>9}{'confidence':>12}{'curr base':>11}{'lift':>8}")
    for r in main_table.itertuples():
        say(f"    {r.rank:>3} {r.pattern:<42}{r.count:>9,}{r.support:>9.2%}{r.confidence:>12.2%}{r.curr_base_rate:>11.2%}{r.lift:>8.4f}")
    all_filings_base = df["reaction_category"].value_counts(normalize=True).reindex(REACTIONS)
    say("  curr base rate among transitions vs among all filings: " + ", ".join(f"{c} {main_table.drop_duplicates('curr_reaction').set_index('curr_reaction').loc[c, 'curr_base_rate']:.2%} / {all_filings_base[c]:.2%}" for c in REACTIONS))
    tie = main_table["count"].iloc[TOP - 1] == main_table["count"].iloc[TOP]
    say(f"  Rank {TOP + 1}: {main_table['pattern'].iloc[TOP]} {main_table['count'].iloc[TOP]:,} ({main_table['support'].iloc[TOP]:.2%}). " + ("TIE at the cut-off." if tie else "No tie at the cut-off."))
    say("  The 2 most frequent patterns as sequential rules:")
    for r in main_table.head(TOP).itertuples():
        say(f"    {r.prev_reaction} => {r.curr_reaction} (next filing): support {r.support:.2%} ({r.count:,}), confidence {r.confidence:.2%} ({r.count:,} / {r.prev_count:,}), "
            f"base rate {r.curr_base_rate:.2%}, lift {r.lift:.4f}")
    best_conf = main_table.sort_values("confidence", ascending=False).iloc[0]
    say(f"  Highest-confidence pattern of all 25, for reference: {best_conf['pattern']} confidence {best_conf['confidence']:.2%}, lift {best_conf['lift']:.4f}, support {best_conf['support']:.2%}.")

    # ---------------------------------------------------------------- 3. robustness
    say("\nSTEP 3: ROBUSTNESS")
    say(f"  A: transitions inside one (ticker, event_category): the previous filing is the company's previous filing of the same category.")
    say(f"  B: split at the median gap of {median_gap:.0f} trading days: short = gap <= {median_gap:.0f}, long = gap > {median_gap:.0f} "
        f"({(main_t['gap_group'] == 'short').mean():.2%} short; the median value itself holds {(main_t['gap_trading_days'] == median_gap).sum():,} transitions).")
    say(f"  Extra: gap >= {MIN_GAP_NO_OVERLAP} trading days, so the two CAR[-1,+1] windows share no trading day. Pairs with gap 0 share day 0 and therefore the same CAR.")
    summaries = pd.DataFrame([summary(variants[name], tables[name], name) for name in variants])
    for s in summaries.to_dict("records"):
        say(f"  {s['variant']}: {s['transitions']:,} transitions, {s['companies']:,} companies, median gap {s['median_gap_trading_days']:.0f}; "
            f"same reaction as previous {s['same_reaction_share']:.2%} (if independent {s['same_reaction_if_independent']:.2%})")
        for i in range(1, TOP + 1):
            say(f"    top {i}: {s[f'pattern_{i}']:<40} support {s[f'pattern_{i}_support']:.2%}  confidence {s[f'pattern_{i}_confidence']:.2%}  lift {s[f'pattern_{i}_lift']:.4f}")
    main_top = list(main_table["pattern"].head(TOP))
    tracked = pd.concat([tables[name][tables[name]["pattern"].isin(main_top)] for name in variants], ignore_index=True)
    say("  The main top-2 patterns in every variant:")
    for r in tracked.itertuples():
        say(f"    {r.variant:<58}{r.pattern:<40} rank {r.rank:>2}  support {r.support:.2%}  confidence {r.confidence:.2%}  lift {r.lift:.4f}")
    gap0 = main_t["gap_trading_days"] == 0
    say(f"  Gap 0 (same reaction day 0): {percent(int(gap0.sum()), len(main_t))} transitions; same reaction in {(main_t.loc[gap0, 'prev_reaction'] == main_t.loc[gap0, 'reaction_category']).mean():.2%} of them.")

    # ---------------------------------------------------------------- 4. within-company shuffle
    say(f"\nSTEP 4: EXTRA CHECK -- WITHIN-COMPANY SHUFFLE ({SHUFFLES} shuffles, seed {SEED})")
    say("  Each company keeps its own reactions, in a random order. What the shuffle still shows comes from companies differing in their reaction mix; what it loses is order.")
    rng = np.random.default_rng(SEED)
    company_code = pd.factorize(df["ticker"])[0]
    has_prev = df["ticker"].eq(df["ticker"].shift(1)).to_numpy()
    reaction = df["reaction_category"].to_numpy()
    shuffled_counts, shuffled_same = [], []
    for _ in range(SHUFFLES):
        order = np.lexsort((rng.random(n), company_code))   # df is sorted by ticker, so this permutes inside each company's block
        assert (company_code[order] == company_code).all()
        r = reaction[order]
        prev, curr = r[:-1][has_prev[1:]], r[1:][has_prev[1:]]
        shuffled_counts.append(pd.crosstab(prev, curr).reindex(index=REACTIONS, columns=REACTIONS, fill_value=0))
        shuffled_same.append((prev == curr).mean())
    expected = sum(shuffled_counts) / SHUFFLES
    spread = np.std([c.to_numpy() for c in shuffled_counts], axis=0)
    shuffle_rows = pd.DataFrame([{"pattern": f"<{a}> -> <{b}>", "observed_count": int(main_table.set_index("pattern").loc[f"<{a}> -> <{b}>", "count"]),
                                  "shuffled_mean_count": expected.loc[a, b], "shuffled_sd": spread[REACTIONS.index(a), REACTIONS.index(b)],
                                  "observed_over_shuffled": main_table.set_index("pattern").loc[f"<{a}> -> <{b}>", "count"] / expected.loc[a, b]}
                                 for a in REACTIONS for b in REACTIONS])
    shuffle_rows = shuffle_rows.set_index("pattern").loc[main_table["pattern"]].reset_index()
    for r in shuffle_rows.head(TOP).itertuples():
        say(f"  {r.pattern}: observed {r.observed_count:,}, shuffled {r.shuffled_mean_count:,.0f} (sd {r.shuffled_sd:.0f}), ratio {r.observed_over_shuffled:.4f}")
    summaries["same_reaction_within_company_shuffle"] = np.where(summaries["variant"] == MAIN, np.mean(shuffled_same), np.nan)   # the shuffle is of all transitions
    same_main = summaries.set_index("variant").loc[MAIN, "same_reaction_share"]
    same_indep = summaries.set_index("variant").loc[MAIN, "same_reaction_if_independent"]
    say(f"  Same reaction as previous: observed {same_main:.2%}; within-company shuffle {np.mean(shuffled_same):.2%} (sd {np.std(shuffled_same):.3%}); if independent of company {same_indep:.2%}.")
    no_overlap_t = variants[NO_OVERLAP]
    same_no_overlap = (no_overlap_t["prev_reaction"] == no_overlap_t["reaction_category"]).mean()

    # ---------------------------------------------------------------- 5. flags
    s = summaries.set_index("variant")
    flags = [
        "Reframing: Phase 2's persistence question compares CAR[-1,+1] with CAR[-1,+5] for the same filing (decay within one event). This task asks a different question: whether one filing's reaction predicts the same company's next filing's reaction (persistence across events).",
        f"Pairs that share reaction days inflate persistence. {percent(int(gap0.sum()), len(main_t))} transitions have the same reaction day 0 and so the same CAR and the same reaction_category by construction; gaps of 1-2 trading days share part of the CAR window. "
        f"Same reaction as previous: {same_main:.2%} over all transitions, {same_no_overlap:.2%} with gap >= {MIN_GAP_NO_OVERLAP}.",
        f"Once pairs that share CAR days are removed, little order effect is left. Same reaction as previous is {same_no_overlap:.2%} with gap >= {MIN_GAP_NO_OVERLAP}; shuffling each company's reactions (which keeps its mix and removes order) gives {np.mean(shuffled_same):.2%}; "
        f"independence from the company gives {same_indep:.2%}. So of the {(same_no_overlap - same_indep) * 100:.2f} points above independence, {(np.mean(shuffled_same) - same_indep) * 100:.2f} are matched by company mix alone (companies that often have Strong reactions keep having them, in any order). "
        "The shuffle covers all transitions, not only the gap >= 3 ones, so this split is approximate.",
        f"Order: filings are sequenced by filing_date as asked, but reactions are measured from report_date. {negative:,} transitions ({negative / len(main_t):.2%}) have the earlier-filed filing's reaction day later than the next filing's.",
        f"lift uses curr's base rate among the transitions in each variant; among all filings the rates differ by at most {(main_table.drop_duplicates('curr_reaction').set_index('curr_reaction')['curr_base_rate'] - all_filings_base).abs().max() * 100:.2f} points.",
        f"Robustness A changes which pairs exist, not just their number: {s.loc[SAME_CATEGORY, 'transitions']:,} transitions (vs {s.loc[MAIN, 'transitions']:,}), and its gaps are longer (median {s.loc[SAME_CATEGORY, 'median_gap_trading_days']:.0f} trading days vs {median_gap:.0f}), and it keeps {percent(int((variants[SAME_CATEGORY]['gap_trading_days'] == 0).sum()), s.loc[SAME_CATEGORY, 'transitions'])} gap-0 pairs (vs {gap0.sum():,} in the main table).",
        "Inherited from Phase 2: filings on the same company and trading day share their CAR; size and volatility are October 2026 snapshots; the universe has survivorship bias.",
    ]
    say("\nFLAGS")
    for i, flag in enumerate(flags, start=1):
        say(f"  {i}. {flag}")

    # ---------------------------------------------------------------- write everything
    data = main_t[["ticker", "event_id", "filing_date", "trading_date", "event_category", "reaction_category", "prev_event_id", "prev_filing_date",
                   "prev_event_category", "prev_reaction", "gap_trading_days", "gap_group", "same_event_category"]].rename(columns={"reaction_category": "curr_reaction"})
    data.to_csv(PROC / "sequential_events.csv", index=False)
    all_patterns = pd.concat(tables.values(), ignore_index=True)
    all_patterns.to_csv(PROC / "sequential_patterns.csv", index=False)
    summaries.to_csv(PROC / "sequential_summary.csv", index=False)
    shuffle_rows.to_csv(PROC / "sequential_shuffle_check.csv", index=False)

    share, count, score = "0.00%", "#,##0", "0.0000"
    fmt = {c: share for c in ("support", "confidence", "curr_base_rate", "same_reaction_share", "same_reaction_if_independent", "same_reaction_within_company_shuffle", *[f"pattern_{i}_{m}" for i in (1, 2) for m in ("support", "confidence")])}
    fmt |= {c: count for c in ("count", "prev_count", "transitions", "companies", "observed_count", "shuffled_mean_count")}
    fmt |= {c: score for c in ("lift", "pattern_1_lift", "pattern_2_lift", "observed_over_shuffled")} | {"shuffled_sd": "0.0", "median_gap_trading_days": "0"}

    def wide(column, label):
        return main_table.pivot(index="prev_reaction", columns="curr_reaction", values=column).reindex(index=REACTIONS, columns=REACTIONS).reset_index().rename(columns={"prev_reaction": label})

    wb = open_workbook()
    write_sheet(wb, "T2_Sequential_Data", data, {"filing_date": "yyyy-mm-dd", "trading_date": "yyyy-mm-dd", "prev_filing_date": "yyyy-mm-dd", "gap_trading_days": "0"},
                {"ticker": 9, "event_id": 12, "prev_event_id": 14, "event_category": 22, "prev_event_category": 22, "curr_reaction": 16, "prev_reaction": 16})
    write_blocks(wb, "T2_Sequential_Results", [
        ("Every variant: size, same-reaction share and its 2 most frequent patterns", summaries, fmt),
        (f"All 25 patterns, all consecutive filings ({len(main_t):,} transitions), by support. Rule prev => curr: confidence = count / prev_count, lift = confidence / curr_base_rate", main_table.drop(columns="variant"), fmt),
        ("Rows behind each pattern (row: previous filing's reaction, column: next filing's)", wide("count", "prev \\ curr (count)"), {r: count for r in REACTIONS}),
        ("Confidence: P(next reaction | previous reaction)", wide("confidence", "prev \\ curr (confidence)"), {r: share for r in REACTIONS}),
        ("Lift: confidence / base rate of the next reaction", wide("lift", "prev \\ curr (lift)"), {r: score for r in REACTIONS}),
        ("The 2 most frequent patterns of the main table, in every variant", tracked, fmt),
        (f"Extra check: observed counts against a within-company shuffle ({SHUFFLES} shuffles, seed {SEED})", shuffle_rows, fmt),
        ("All 25 patterns in every variant", all_patterns, fmt),
    ], {"A": 62, "B": 40, "C": 18, "D": 18, **{letter: 15 for letter in "EFGHIJKLMNOPQRS"}})
    notes = [
        f"Rows used: {n:,} filings from {df['ticker'].nunique():,} companies, the same table as Task 1 (Phase 2 events + market_reactions + companies, minus {unreliable} rows with price_data_unreliable).",
        "Table: per company, the filings sorted by filing_date (ties by event_id). Each filing is paired with the company's previous filing: prev_reaction = groupby(ticker)['reaction_category'].shift(1). A company's first filing has no previous one and is dropped. "
        f"That leaves {len(main_t):,} transitions, in T2_Sequential_Data.",
        "Tool: Python (pandas). Each pattern <A> -> <B> is counted exactly over all transitions (a sequence of length 2, so every pattern of that length is counted; nothing is pruned). "
        "support = count(A then B) / all transitions; rule A => B: confidence = count(A then B) / count(A first); lift = confidence / share of B among the next filings.",
        "Top 2 patterns: " + "; ".join(f"{r.pattern} support {r.support:.2%}, confidence {r.confidence:.2%}, lift {r.lift:.4f}" for r in main_table.head(TOP).itertuples()) + ".",
        f"Robustness A: the same, with the previous filing taken inside one (ticker, event_category): {s.loc[SAME_CATEGORY, 'transitions']:,} transitions.",
        f"Robustness B: trading-day gap between the two filings' reaction day 0 on the market calendar of Tier 0 (every distinct trading date in the data), split at its median of {median_gap:.0f}: "
        f"short = gap <= {median_gap:.0f} ({s.loc[SHORT, 'transitions']:,}), long = gap > {median_gap:.0f} ({s.loc[LONG, 'transitions']:,}).",
        *[f"{v}: top patterns {r['pattern_1']} (support {r['pattern_1_support']:.2%}, confidence {r['pattern_1_confidence']:.2%}, lift {r['pattern_1_lift']:.4f}) and {r['pattern_2']} (support {r['pattern_2_support']:.2%}, confidence {r['pattern_2_confidence']:.2%}, lift {r['pattern_2_lift']:.4f}); same reaction as previous {r['same_reaction_share']:.2%} vs {r['same_reaction_if_independent']:.2%} if independent."
          for v, r in s.iterrows()],
        f"Extra check (not in the brief): shuffling the reactions inside each company {SHUFFLES} times keeps each company's mix and removes order. Same reaction as previous: observed {same_main:.2%}, shuffled {np.mean(shuffled_same):.2%}, independent {same_indep:.2%}.",
        "Values are outputs of data_phase3/sequential_events.py, stored as values rather than formulas. processed/sequential_patterns.csv holds all 25 patterns of every variant.",
        "Issues found:",
        *[f"{i}. {flag}" for i, flag in enumerate(flags, start=1)],
    ]
    write_notes(wb, "T2_Sequential_Notes", notes)
    save_workbook(wb)
    say("\nWROTE")
    say(f"  processed/sequential_events.csv ({len(data):,} rows), sequential_patterns.csv ({len(all_patterns)} rows), sequential_summary.csv, sequential_shuffle_check.csv")
    say("  220227286.xlsx: T2_Sequential_Data, T2_Sequential_Results, T2_Sequential_Notes")
    (PROC / "sequential_events_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
