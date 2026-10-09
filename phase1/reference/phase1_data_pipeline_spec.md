# Implementation Spec: Corporate Events & Abnormal Stock Returns Dataset
### For EECS 4412 Phase 1 — hand this whole file to your coding agent as the prompt/spec

## 0. Context

Course project (solo). Research question: **which characteristics of a corporate event (category, timing, disclosure type) are associated with the direction and magnitude of a company's abnormal stock return?**

Three entities are required (course minimums: ≥3 entities, ≥7 attributes each spanning nominal/ordinal/interval/ratio, ≥50 instances each). This spec defines exactly what to build. A companion file `companies_sample_57.csv` is provided — **use it as-is, do not re-sample**, so the report and the data stay consistent.

**Your job (the coding agent):** write a Python script (or a couple of scripts) that fetches real data from SEC EDGAR and Yahoo Finance, computes the derived fields below, and outputs exactly the three CSVs described in sections 2-4, plus a short run log. No placeholder/synthetic data — every row must trace back to a real filing or real price data.

**Environment note:** you likely have normal, unrestricted internet access (unlike the sandbox this spec was drafted in) — use `requests` and `yfinance` directly rather than any manual/browser workaround.

---

## 1. Inputs provided

- `companies_sample_57.csv` — 57 S&P 500 companies, stratified by GICS sector, columns: `Symbol, Security, GICS Sector, GICS Sub-Industry, Headquarters Location, Date added, CIK, Founded, DeepDive`.
- `DeepDive == 'Y'` flags **15 companies** that need full event + price treatment: AAPL, MSFT, JNJ, UNH, JPM, GS, AMZN, GOOGL, META, BA, PG, XOM, NEE, LIN, AMT.
- The other 42 rows only need to be enriched with fundamentals (section 2) — no filings/price work needed for them.

---

## 2. Entity 1 — `companies.csv` (target: 57 rows, one per company)

Start from `companies_sample_57.csv` and add fundamentals via `yfinance` (`yf.Ticker(symbol).info` or `.get_info()`). Handle missing/rate-limited lookups gracefully (retry with backoff, sleep ~0.5-1s between tickers, log any ticker that fails after 3 retries rather than crashing the whole run).

Final columns:

| Column | Type | Source / derivation |
|---|---|---|
| `ticker` | nominal | from input CSV (`Symbol`) |
| `company_name` | nominal | from input CSV (`Security`) |
| `sector` | nominal | from input CSV (`GICS Sector`) |
| `sub_industry` | nominal | from input CSV (`GICS Sub-Industry`) |
| `exchange` | nominal | `yfinance` `.info['exchange']` |
| `cik` | nominal | from input CSV (`CIK`) |
| `founded_year` | interval | from input CSV (`Founded`) — **strip parenthetical/compound values** (e.g. `"2000 (1799 / 1871)"` → keep the first 4-digit year only; document this cleaning step) |
| `size_category` | ordinal | derive AFTER fetching market cap: split `market_cap` into quartiles across all 57 rows → label Small/Mid/Large/Mega-cap. Do not hardcode thresholds — compute from the sample itself (`pandas.qcut`). |
| `market_cap` | ratio | `yfinance` `.info['marketCap']` |
| `revenue_ttm` | ratio | `yfinance` `.info['totalRevenue']` |
| `employees` | ratio | `yfinance` `.info['fullTimeEmployees']` |

That's 11 columns (well over the 7 minimum), one of each required type plus extras.

---

## 3. Entity 2 — `events.csv` (target: 80-250 rows total across the 15 deep-dive companies)

**Source:** SEC EDGAR submissions API — `https://data.sec.gov/submissions/CIK{10-digit-zero-padded-cik}.json` for each of the 15 `DeepDive=='Y'` companies.

**Required header (SEC will reject requests without a real identifying User-Agent):**
```python
headers = {"User-Agent": "EECS4412 Student Project your-real-email@yorku.ca"}
```
Add a small sleep (~0.2s) between requests. Cache each raw JSON response to disk before parsing (so re-runs don't re-hit the API).

**Filtering logic:**
1. Parse `filings.recent` (arrays: `form`, `filingDate`, `reportDate`, `items`, `accessionNumber`, `primaryDocument`).
2. Keep only rows where `form == '8-K'`.
3. Keep only rows where `reportDate` (fall back to `filingDate` if `reportDate` is blank) falls in **2025-09-01 to 2026-09-17**.
4. If a company yields very few filings in that window, note it in the run log — do not silently drop the company. If the TOTAL across all 15 companies comes out under 60, widen the window to 2025-01-01–2026-09-17 for everyone and re-run. If it comes out over ~300, narrow to the most recent 8 months instead. Log whichever window was actually used.

**Category mapping** (apply to each filing's `items` list — a filing can have multiple items; use the first/primary item code for `event_category`, but keep the raw `item_codes` too):

| event_category | 8-K item codes |
|---|---|
| Financial/Earnings | 2.02, 2.06, 1.03 |
| M&A/Contracts | 1.01, 1.02, 2.01, 5.01 |
| Management/Governance | 5.02, 4.01, 5.03 |
| Legal/Regulatory | 3.01, 4.02 |
| Capital/Financing | 2.03, 7.01 |
| Other | 8.01, 2.05, anything unmapped |

Final columns:

| Column | Type | Derivation |
|---|---|---|
| `event_id` | nominal | generate sequential, e.g. `EVT0001` |
| `ticker` | nominal | which company |
| `accession_number` | nominal | from SEC data (unique filing ID, useful as a real-world reference/citation) |
| `item_codes` | nominal | raw comma-joined items string from SEC |
| `event_category` | nominal | via mapping table above |
| `filing_date` | interval | `filingDate` |
| `report_date` | interval | `reportDate` (fallback to `filingDate` if blank) |
| `filing_timeliness` | ordinal | days = (filing_date − report_date); bucket: `On-time` (0-4 days, matches SEC's 4-business-day rule), `Slightly Late` (5-10), `Late` (>10) |
| `num_items_disclosed` | ratio | count of items in `item_codes` |

That's 9 columns.

---

## 4. Entity 3 — `market_reactions.csv` (one row per row in `events.csv`, so same target ~80-250)

**Source:** `yfinance` daily price history for each of the 15 tickers AND the S&P 500 index (`^GSPC`), covering a buffer of at least 15 trading days before the earliest event date through 15 trading days after the latest event date in `events.csv`.

```python
import yfinance as yf
px = yf.download(ticker, start=..., end=..., auto_adjust=True)
idx = yf.download("^GSPC", start=..., end=..., auto_adjust=True)
```

**For each event** (matched on `report_date`, or the next trading day if `report_date` falls on a weekend/holiday — call this `t0`):

- `daily_return` = (Close[t0] − Close[t0−1]) / Close[t0−1]
- `market_return` = same formula on `^GSPC` for the same two dates
- `abnormal_return` = `daily_return` − `market_return` (market-adjusted model — simplest defensible event-study approach, fine for this course)
- `car_minus1_plus1` = sum of daily abnormal returns over [t0−1, t0, t0+1]
- `car_minus1_plus5` = sum of daily abnormal returns over [t0−1 … t0+5]
- `volume_change_pct` = (Volume[t0] − mean(Volume[t0−10 … t0−1])) / mean(Volume[t0−10 … t0−1])
- `reaction_category` (ordinal, derived from `car_minus1_plus1`): `Strong Negative` (< −3%), `Negative` (−3% to −0.5%), `Neutral` (−0.5% to +0.5%), `Positive` (+0.5% to +3%), `Strong Positive` (> +3%)

If price data is missing for an event date (e.g. ticker delisted-and-relisted gap, data hole), drop that event row from both `events.csv` and `market_reactions.csv` and log it — don't fabricate a value.

Final columns:

| Column | Type |
|---|---|
| `event_id` | nominal (FK to events.csv) |
| `trading_date` | interval |
| `daily_return` | ratio |
| `market_return` | ratio |
| `abnormal_return` | ratio |
| `car_minus1_plus1` | ratio |
| `car_minus1_plus5` | ratio |
| `volume_change_pct` | ratio |
| `reaction_category` | ordinal |

9 columns.

---

## 5. Run log

Print/save a short summary: rows written per file, date window actually used for events, any tickers/events dropped and why, total runtime. This is what I'll use to sanity-check the data before we write it up.

---

## 6. What NOT to do

- Don't invent or interpolate values when a real lookup fails — log and drop/skip instead.
- Don't re-sample the 57 companies or change the 15 deep-dive tickers — the report text is being written around this exact set.
- Don't build the statistical-analysis CSV (`T1.csv`) — that's a separate, smaller task done after this data lands, once the actual distributions are known.

## 7. Handoff

Once you have `companies.csv`, `events.csv`, `market_reactions.csv`, and the run log — send all four back into the chat with Claude (the one drafting the EECS 4412 report). That's where the visualizations, statistical analysis, and Word document get built.
