"""
EECS 4412 Phase 2, Task 1 -- Principal Component Analysis on the Companies entity.

  1. report missing values, then drop incomplete companies (nothing is imputed)
     and the 13 companies with spliced-looking prices in the volatility window
  2. encode: size_category -> 1..4, log1p market_cap/revenue_ttm/employees,
     one-hot sector and exchange; trailing_volatility_1y stays as stored
  3. z-score every encoded column
  4. fit PCA on all components, keep the top 3
  5. write the scores, the explained variance and the loadings, and report anything
     about the 3,900-company table that differs from Phase 1's 61 companies

Run with the project venv (needs sklearn + openpyxl):  ..\\..\\.venv\\Scripts\\python.exe pca_companies.py
Writes, in processed/:  pca_companies_scores.csv, pca_companies_explained_variance.csv,
                        pca_companies_loadings.csv, pca_companies_log.txt
                        220227286.xlsx (sheets T1_PCA_*; any other sheets are kept)
Writes, in raw/:        pca_dropped_companies.csv (which companies were left out, and why)
Charts come from ../charts2/pca_charts.py, which reads the files above.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROC = HERE / "processed"
RAW = HERE / "raw"
PHASE1 = HERE.parents[1] / "phase1"
WORKBOOK = PROC / "220227286.xlsx"

SIZE_ORDER = {"Small": 1, "Mid": 2, "Large": 3, "Mega-cap": 4}
LOG_COLUMNS = ["market_cap", "revenue_ttm", "employees"]
VOLATILITY = "trailing_volatility_1y"
NOMINAL = ["sector", "exchange"]
DIMENSIONS = ["size_category", *LOG_COLUMNS, VOLATILITY, *NOMINAL]
N_KEEP = 3

# Inside the trailing-volatility window these 13 have a +300% day and a -75% day within 5 sessions.
# Checked on 2026-11-01 against Yahoo's split records and current prices.
# Bad source data: closes switch between split-adjusted and unadjusted before a reverse split.
BAD_PRICE_DATA = {
    "NFE": "1-for-50 split on 2026-09-14", "BURU": "1-for-40 split on 2026-09-02",
    "OPTT": "1-for-30 split on 2026-09-14", "GTBP": "1-for-25 split on 2026-09-08",
    "BRTX": "1-for-20 split on 2026-09-08", "MTNB": "1-for-15 split on 2026-09-28",
    "NXXT": "1-for-10 split on 2026-09-14", "VWAV": "1-for-20 split on 2026-09-22",
    "BTLN": "1-for-8 split on 2026-09-28", "WHLR": "1-for-9 split on 2026-09-22",
    "LRHC": "1-for-6 split on 2026-09-08; split record and jump sizes only, Yahoo's current prices do not cover the dates",
}
# Same pattern, but the prices look real. Excluded as instructed; delete a line here to keep a company.
PATTERN_ONLY = {
    "CPHI": "no split on record; press reports match the close ($8.10, +790%) and volume (93.6M) on 2026-07-21",
    "PLAG": "no split on record; prices match Yahoo's current data, 213.7M shares traded on 2026-08-11; no press report found",
}
EXCLUDED = {**BAD_PRICE_DATA, **PATTERN_ONLY}
# Same defect suspected but not in the 13: a single jump of about the split size on collapsing volume.
# They stay in the PCA; main reports how much they matter.
SUSPECTED_ONE_WAY = ["IPDN", "DUKR", "OPAD"]

HEADER_FILL = PatternFill(start_color="2A78D6", end_color="2A78D6", fill_type="solid")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial")

log_lines = []


def say(text=""):
    """Print a line and keep it for pca_companies_log.txt."""
    print(text)
    log_lines.append(text)


def log1p_clipped(values):
    """The rule of log_or_zero in ../network/tier0_prepare_data.py: negatives become 0, then log1p."""
    return np.log1p(values.clip(lower=0))


def signed_log1p(values):
    """Keeps negative revenue negative. Only for the sensitivity check in main."""
    return np.sign(values) * np.log1p(values.abs())


def encode(df, transform=log1p_clipped, nominal=NOMINAL):
    """The 7 dimensions as numbers. The arguments are only changed by the sensitivity checks in main."""
    x = pd.DataFrame({"size_category": df["size_category"].map(SIZE_ORDER).astype(float)})
    for col in LOG_COLUMNS:
        x[col] = transform(df[col].astype(float))
    x[VOLATILITY] = df[VOLATILITY].astype(float)
    one_hots = [pd.get_dummies(df[col], prefix=col, prefix_sep="=").astype(float) for col in nominal]
    return pd.concat([x, *one_hots], axis=1)


def fit(x):
    """z-score, then PCA on every component. svd_solver='full' gives the same answer every run."""
    z = StandardScaler().fit_transform(x)
    pca = PCA(svd_solver="full").fit(z)
    return pca, pca.transform(z)


def top1pct_share(scores):
    """Share of a component's squared scores held by its most extreme 1% of companies."""
    squared = np.sort(scores ** 2)[::-1]
    return squared[:len(squared) // 100].sum() / squared.sum()


def top_loadings(loadings, pc, n=5):
    ranked = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
    return ", ".join(f"{name} {value:+.2f}" for name, value in ranked.head(n).items())


def write_sheet(wb, name, df, formats=None, widths=None):
    """Replace one sheet, in the same style as the Phase 1 workbook."""
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([v.item() if isinstance(v, np.generic) else v for v in row])
    for cell in ws[1]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    for col_idx, col in enumerate(df.columns, start=1):
        fmt = (formats or {}).get(col)
        for row_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row_idx, col_idx)
            cell.font = BODY_FONT
            if fmt:
                cell.number_format = fmt
        letter = ws.cell(1, col_idx).column_letter
        ws.column_dimensions[letter].width = (widths or {}).get(col, max(12, len(col) + 3))
    ws.freeze_panes = "A2"
    return ws


def main():
    say("EECS 4412 Phase 2, Task 1: PCA on the Companies entity")
    say("=" * 70)
    read = dict(keep_default_na=False, na_values=[""])   # real tickers such as "NA" stay strings
    companies = pd.read_csv(PROC / "companies.csv", **read)
    phase1 = pd.read_csv(PHASE1 / "data" / "processed" / "companies.csv", **read)
    say(f"Loaded processed/companies.csv: {len(companies):,} companies, {companies.shape[1]} columns "
        f"(Events and Market_Reactions have 9 each, so Companies has the most dimensions).")

    # ---------------------------------------------------------------- dimensions
    say("\nDIMENSIONS")
    say("  size_category (ordinal); market_cap, revenue_ttm, employees, trailing_volatility_1y (ratio);")
    say("  sector, exchange (nominal). ticker is kept as the row label only.")

    # ---------------------------------------------------------------- 1. missing values, before anything else
    say("\nSTEP 1: MISSING VALUES (reported first; incomplete companies are dropped, nothing is imputed)")
    for col in DIMENSIONS:
        say(f"  {col:<24}{companies[col].isna().sum():>5} missing")
    is_complete = companies[DIMENSIONS].notna().all(axis=1)
    is_excluded = companies["ticker"].isin(EXCLUDED)
    assert is_excluded.sum() == len(EXCLUDED)
    dropped = companies[~is_complete | is_excluded].copy()
    dropped["missing_columns"] = dropped[DIMENSIONS].isna().apply(lambda row: ",".join(row.index[row]), axis=1)
    dropped["price_exclusion"] = dropped["ticker"].map(EXCLUDED)
    dropped[["ticker", "company_name", "sector", "size_category", "missing_columns", "price_exclusion"]].to_csv(
        RAW / "pca_dropped_companies.csv", index=False)
    df = companies[is_complete & ~is_excluded].reset_index(drop=True)
    assert df["ticker"].is_unique and df[DIMENSIONS].notna().all().all()
    already_gone = sorted(companies.loc[is_excluded & ~is_complete, "ticker"])
    say(f"  FISV is confirmed incomplete: missing {dropped.set_index('ticker').loc['FISV', 'missing_columns']}.")
    say(f"  Dropped {(~is_complete).sum():,} companies with at least one blank.")
    say("\n  PRICE EXCLUSIONS (spliced-looking prices inside the 1-year volatility window)")
    say(f"  {len(BAD_PRICE_DATA)} have bad source data (mixed split-adjusted and unadjusted prices):")
    for ticker, evidence in BAD_PRICE_DATA.items():
        say(f"    {ticker:<6}{evidence}")
    say(f"  {len(PATTERN_ONLY)} match the pattern but their prices look real; excluded as instructed:")
    for ticker, evidence in PATTERN_ONLY.items():
        say(f"    {ticker:<6}{evidence}")
    say(f"  {len(already_gone)} of the {len(EXCLUDED)} already had a blank ({', '.join(already_gone)}), so this removes "
        f"{(is_complete & is_excluded).sum()} more.")
    say(f"  All of them are listed in raw/pca_dropped_companies.csv.")
    say(f"\n  COMPANY COUNT USED: {len(df):,} of {len(companies):,} ({len(df) / len(companies):.1%})")
    say("  Dropped share by size:   " + ", ".join(
        f"{s} {(dropped['size_category'] == s).sum() / (companies['size_category'] == s).sum():.1%}" for s in SIZE_ORDER))
    worst = (dropped["sector"].value_counts() / companies["sector"].value_counts()).dropna().sort_values(ascending=False)
    say("  Dropped share by sector, highest three: " + ", ".join(f"{s} {v:.1%}" for s, v in worst.head(3).items()))

    # ---------------------------------------------------------------- 2-3. encode and standardise
    x = encode(df)
    say("\nSTEP 2: ENCODING")
    n_negative = (df["revenue_ttm"] < 0).sum()
    say("  size_category: Small=1, Mid=2, Large=3, Mega-cap=4")
    say("  market_cap, revenue_ttm, employees: log1p, using Tier 0's rule (negative values become 0 first).")
    say(f"  This changes {n_negative} companies with negative revenue_ttm; {(df['revenue_ttm'] == 0).sum()} more report exactly 0.")
    say("  trailing_volatility_1y: as stored, not logged.")
    say("  " + " | ".join(f"{col}: one-hot, {df[col].nunique()} columns" for col in NOMINAL))
    say(f"  7 dimensions become {x.shape[1]} encoded columns; loadings are reported across those {x.shape[1]}.")
    say("\nSTEP 3: STANDARDISATION")
    say("  Every encoded column, one-hot columns included: z = (value - mean) / population standard deviation.")
    say("  Each column therefore has variance 1 and total variance equals the number of columns.")

    # ---------------------------------------------------------------- 4. PCA
    pca, scores = fit(x)
    names = [f"PC{i + 1}" for i in range(x.shape[1])]
    keep = names[:N_KEEP]
    variance = pd.DataFrame({
        "component": names,
        "eigenvalue": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        "kept": [name in keep for name in names],
    })
    loadings = pd.DataFrame(pca.components_[:N_KEEP].T, index=x.columns, columns=keep)
    score_table = pd.DataFrame(scores[:, :N_KEEP], columns=keep)
    score_table.insert(0, "ticker", df["ticker"])

    say("\nSTEP 4: PCA (scikit-learn PCA, svd_solver='full')")
    say("  component  eigenvalue  explained  cumulative")
    for row in variance.itertuples():
        say(f"  {row.component:<9}{row.eigenvalue:>11.4f}{row.explained_variance_ratio:>11.2%}{row.cumulative_explained_variance:>12.2%}")
    say(f"  Kept PC1-PC3: {variance['cumulative_explained_variance'].iloc[N_KEEP - 1]:.2%} of the variance.")
    n_zero = (pca.explained_variance_ratio_ < 1e-12).sum()
    say(f"  The last {n_zero} components are zero because each one-hot group sums to 1, which removes one")
    say(f"  degree of freedom per group: {x.shape[1]} columns, rank {x.shape[1] - n_zero}.")
    say("\n  Largest loadings (a loading here = the unit-length eigenvector weight):")
    for pc in keep:
        say(f"  {pc}: {top_loadings(loadings, pc)}")
        extreme = score_table.set_index("ticker")[pc].abs().sort_values(ascending=False).head(3)
        say(f"       most extreme: {', '.join(f'{t} {v:.1f}' for t, v in extreme.items())}"
            f" | top 1% of companies hold {top1pct_share(score_table[pc].to_numpy()):.0%} of its squared scores")

    # ---------------------------------------------------------------- which colouring separates better
    two_d = score_table[["PC1", "PC2"]].to_numpy()
    sil_sector = silhouette_score(two_d, df["sector"])
    sil_size = silhouette_score(two_d, df["size_category"])
    say("\nSEPARATION IN THE PC1-PC2 PLANE (silhouette, -1 to +1; near 0 = groups overlap)")
    say(f"  size_category {sil_size:+.3f} | sector {sil_sector:+.3f} -> "
        f"{'size_category' if sil_size > sil_sector else 'sector'} separates better; neither forms clean clusters.")

    # ---------------------------------------------------------------- sensitivity (not part of the deliverable)
    say("\nSENSITIVITY CHECKS (reported only; the deliverable above follows the requested preprocessing)")
    pattern_only_kept = companies[is_complete & ~companies["ticker"].isin(BAD_PRICE_DATA)]
    logged_volatility = encode(df)
    logged_volatility[VOLATILITY] = np.log1p(logged_volatility[VOLATILITY])
    checks = {
        f"keeping {' and '.join(PATTERN_ONLY)}": encode(pattern_only_kept),
        f"also excluding {', '.join(SUSPECTED_ONE_WAY)}": encode(df[~df["ticker"].isin(SUSPECTED_ONE_WAY)]),
        "log1p on trailing_volatility_1y too": logged_volatility,
        "without log1p": encode(df, transform=lambda values: values),
        "negative revenue kept negative (signed log)": encode(df, transform=signed_log1p),
        "without the single PNK company (WORX)": encode(df[df["exchange"] != "PNK"]),
    }
    check_shares = {}
    for name, encoded in checks.items():
        check_pca, check_scores = fit(encoded)
        check_shares[name] = check_pca.explained_variance_ratio_[:N_KEEP].sum()
        say(f"  {name}: {encoded.shape[1]} columns, top 3 explain {check_shares[name]:.2%}; "
            f"top 1% hold {top1pct_share(check_scores[:, 0]):.0%} of PC1's squared scores.")

    # ---------------------------------------------------------------- 5. differences from Phase 1
    tiers = phase1[["ticker"]].merge(df[["ticker", "size_category"]], on="ticker")
    tiers_all = phase1[["ticker", "size_category"]].merge(companies[["ticker", "size_category"]], on="ticker", suffixes=("_p1", "_p2"))
    lost = sorted(set(phase1["ticker"]) - set(df["ticker"]))
    ranges = lambda t: f"${t['market_cap'].min():,.0f} to ${t['market_cap'].max():,.0f}"
    flags = [
        f"Scale: {len(phase1)} companies became {len(companies):,} in the table and {len(df):,} here. All {len(phase1)} Phase 1 tickers "
        f"are in the table; {len(tiers)} survive the drop. Lost: {', '.join(lost)}.",
        "founded_year exists in Phase 1 only and is not used.",
        f"size_category changed meaning: quartiles of {len(phase1)} S&P large caps in Phase 1, quartiles of the full census now. "
        f"All {(tiers_all['size_category_p2'] == 'Mega-cap').sum()} Phase 1 companies are Mega-cap on the new scale; only "
        f"{(tiers_all['size_category_p1'] == tiers_all['size_category_p2']).sum()} keep their old tier.",
        f"market_cap range: {ranges(phase1)} in Phase 1, {ranges(companies)} now.",
        f"Skew is far heavier: market_cap {phase1['market_cap'].skew():.1f} in Phase 1, {companies['market_cap'].skew():.1f} now. "
        f"log1p brings it to {np.log1p(companies['market_cap']).skew():.1f}.",
        "sector is Yahoo's taxonomy now, GICS in Phase 1 (e.g. 'Healthcare' vs 'Health Care'), so labels do not join directly.",
        f"trailing_volatility_1y does not exist in Phase 1. After the exclusions it still reaches {df[VOLATILITY].max():.1%} "
        f"({df.loc[df[VOLATILITY].idxmax(), 'ticker']}), with skew {df[VOLATILITY].skew():.1f}. "
        f"{', '.join(SUSPECTED_ONE_WAY)} show a similar one-way jump and remain in the data.",
        f"exchange has {companies['exchange'].nunique()} codes now ({', '.join(sorted(companies['exchange'].unique()))}), "
        f"{phase1['exchange'].nunique()} in Phase 1. PNK is one company (WORX), so its z-scored column is one extreme value.",
        f"Blanks: revenue_ttm {phase1['revenue_ttm'].isna().sum()} and employees {phase1['employees'].isna().sum()} in Phase 1; "
        f"{companies['revenue_ttm'].isna().sum()} and {companies['employees'].isna().sum()} now. Phase 1 has employees for AZO and MU, "
        "but those older values were not copied across, so every company follows one rule.",
        f"Zero or negative revenue_ttm: {(phase1['revenue_ttm'] <= 0).sum()} in Phase 1, {(companies['revenue_ttm'] <= 0).sum()} now. "
        f"Among the companies used, {n_negative} negative values become 0 under the Tier 0 log rule.",
        f"Dropping is not random: small companies and Healthcare are dropped most, so the {len(df):,} lean larger than the full census.",
        f"With {x.shape[1]} unit-variance columns, {x.shape[1] - 2 - len(LOG_COLUMNS)} of them one-hot, three components can only reach a low share "
        f"({variance['cumulative_explained_variance'].iloc[N_KEEP - 1]:.1%}); this reflects the encoding as well as the data.",
    ]
    say("\nDIFFERENCES FROM PHASE 1'S 61-COMPANY ASSUMPTIONS")
    for i, flag in enumerate(flags, start=1):
        say(f"  {i:>2}. {flag}")

    # ---------------------------------------------------------------- write everything
    score_table.to_csv(PROC / "pca_companies_scores.csv", index=False, float_format="%.6f")
    variance.to_csv(PROC / "pca_companies_explained_variance.csv", index=False, float_format="%.6f")
    loadings.rename_axis("encoded_column").reset_index().to_csv(PROC / "pca_companies_loadings.csv", index=False, float_format="%.6f")

    wb = load_workbook(WORKBOOK) if WORKBOOK.exists() else Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]    # the empty sheet a new Workbook starts with
    number = "0.0000"
    percent = "0.00%"
    write_sheet(wb, "T1_PCA_Companies", score_table, {pc: number for pc in keep})
    write_sheet(wb, "T1_PCA_Variance", variance,
                {"eigenvalue": number, "explained_variance_ratio": percent, "cumulative_explained_variance": percent},
                {"explained_variance_ratio": 27, "cumulative_explained_variance": 32})
    write_sheet(wb, "T1_PCA_Loadings", loadings.rename_axis("encoded_column").reset_index(),
                {pc: number for pc in keep}, {"encoded_column": 36})
    notes = pd.DataFrame({"notes": [
        f"Company count used: {len(df):,} of {len(companies):,}. Source: processed/companies.csv.",
        "Dimensions: size_category, market_cap, revenue_ttm, employees, sector, exchange, trailing_volatility_1y.",
        "Missing values: companies with any blank were dropped, none imputed (raw/pca_dropped_companies.csv).",
        f"Price exclusions: {len(EXCLUDED)} companies have a +300% day and a -75% day within 5 sessions inside the 1-year volatility window. "
        f"{len(already_gone)} of them already had a blank ({', '.join(already_gone)}), so this removes {(is_complete & is_excluded).sum()} more. "
        "Checked on 2026-11-01 against Yahoo's split records and current prices.",
        f"Bad source data, mixed split-adjusted and unadjusted prices ({len(BAD_PRICE_DATA)}): "
        + "; ".join(f"{ticker}: {evidence}" for ticker, evidence in BAD_PRICE_DATA.items()) + ".",
        f"Same pattern, but the prices look real; excluded on the project owner's instruction, not as bad data ({len(PATTERN_ONLY)}): "
        + "; ".join(f"{ticker}: {evidence}" for ticker, evidence in PATTERN_ONLY.items()) + ".",
        f"Not excluded: {', '.join(SUSPECTED_ONE_WAY)} show a single jump of about the split size on collapsing volume, so the same defect is suspected.",
        "Encoding: size_category Small=1, Mid=2, Large=3, Mega-cap=4; sector and exchange one-hot.",
        f"Log scaling: log1p of market_cap, revenue_ttm and employees, using Tier 0's rule; {n_negative} negative revenue values become 0 first. "
        "trailing_volatility_1y is not logged.",
        "Standardisation: z-score of every encoded column, including the one-hot columns.",
        "PCA: scikit-learn, svd_solver='full'. A component's sign is arbitrary. Loadings are eigenvector weights.",
        "Values are outputs of data_phase2/pca_companies.py, stored as values rather than formulas.",
        f"Sensitivity: the top 3 explain {variance['cumulative_explained_variance'].iloc[N_KEEP - 1]:.2%} here; "
        + "; ".join(f"{share:.2%} {name}" for name, share in check_shares.items()) + ". Not used in these sheets.",
        "Differences from Phase 1:",
        *[f"{i}. {flag}" for i, flag in enumerate(flags, start=1)],
    ]})
    ws = write_sheet(wb, "T1_PCA_Notes", notes, widths={"notes": 120})
    for row in ws.iter_rows(min_row=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(WORKBOOK)

    say("\nWROTE")
    say(f"  processed/pca_companies_scores.csv ({len(score_table):,} rows), pca_companies_explained_variance.csv, pca_companies_loadings.csv")
    say(f"  processed/{WORKBOOK.name}: T1_PCA_Companies, T1_PCA_Variance, T1_PCA_Loadings, T1_PCA_Notes")
    say(f"  raw/pca_dropped_companies.csv ({len(dropped):,} rows)")
    (PROC / "pca_companies_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
