"""Shared data loading for the Phase 2 chart scripts that split companies by style and sector."""
import numpy as np
import pandas as pd

from chart_style import DATA

CATEGORY_ORDER = ["Financial/Earnings", "Management/Governance", "M&A/Contracts", "Other",
                  "Legal/Regulatory", "Capital/Financing", "Security Incident"]
SHORT = {"Financial/Earnings": "Earnings", "Management/Governance": "Governance", "M&A/Contracts": "M&A",
         "Other": "Other", "Legal/Regulatory": "Legal", "Capital/Financing": "Financing",
         "Security Incident": "Cyber"}


def slug(sector):
    return sector.lower().replace(" ", "_")


def load_events():
    """Events joined to reactions and company attributes, plus derived columns:
    car1/car5/drift in %, primary item and section, style (calm/volatile), and the
    same company's next filing (strictly later report date)."""
    comp = pd.read_csv(DATA / "companies.csv", keep_default_na=False, na_values=[""])
    ev = pd.read_csv(DATA / "events.csv", keep_default_na=False, na_values=[""],
                     parse_dates=["report_date", "filing_date"])
    mr = pd.read_csv(DATA / "market_reactions.csv", parse_dates=["trading_date"])
    df = ev.merge(mr, on="event_id").merge(
        comp[["ticker", "sector", "size_category", "volatility_tier", "trailing_volatility_1y"]], on="ticker")
    df["car1"] = df["car_minus1_plus1"] * 100
    df["car5"] = df["car_minus1_plus5"] * 100
    df["drift"] = df["car5"] - df["car1"]                      # abnormal return over days +2..+5
    items = df["item_codes"].fillna("").str.split(",")
    df["primary"] = items.apply(lambda l: ([i for i in l if i != "9.01"] or l or [""])[0])
    df["section"] = df["primary"].str[0]
    df["style"] = np.where(df["volatility_tier"].isin(["Low", "Medium"]), "calm", "volatile")
    return comp, add_next_filing(df)


def add_next_filing(df):
    """For each event, the company's next filing on a strictly later date. Same-day filings
    share one price reaction, so they are never each other's follow-up."""
    df = df.sort_values(["ticker", "report_date", "accession_number"]).reset_index(drop=True)
    n = len(df)
    nxt = np.full(n, -1)
    for _, idx in df.groupby("ticker", sort=False).indices.items():
        dates = df["report_date"].to_numpy()[idx]
        pos = np.searchsorted(dates, dates, side="right")      # first row with a later date
        ok = pos < len(idx)
        nxt[idx[ok]] = idx[pos[ok]]
    has = nxt >= 0
    take = np.where(has, nxt, 0)
    df["next_category"] = np.where(has, df["event_category"].to_numpy()[take], None)
    df["next_car1"] = np.where(has, df["car1"].to_numpy()[take], np.nan)
    df["next_gap_days"] = np.where(has, (df["report_date"].to_numpy()[take] - df["report_date"].to_numpy())
                                   / np.timedelta64(1, "D"), np.nan)
    t0 = df["trading_date"].to_numpy().astype("datetime64[D]")
    gap_b = np.busday_count(t0, t0[take]).astype(float)        # trading days between event days (weekdays)
    df["next_gap_tdays"] = np.where(has, gap_b, np.nan)
    return df


def binned_median(x, y, edges, min_n=30):
    """Median of y in each x bin; every point whose x is in the bin counts (y is never clipped)."""
    g = pd.DataFrame({"x": x, "y": y}).groupby(pd.cut(x, edges), observed=True)["y"].agg(["median", "size"])
    g = g[g["size"] >= min_n]
    return np.array([iv.mid for iv in g.index]), g["median"].to_numpy(), g["size"].to_numpy()
