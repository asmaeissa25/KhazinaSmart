"""
Universal grocery data adapter.
Auto-detects date, sales, product/category, and store columns from any CSV.
Normalizes to a standard internal format for the universal model.
"""
import pandas as pd
import numpy as np
import re


_DATE_HINTS    = ["date", "week", "period", "time", "day", "month"]
_SALES_HINTS   = ["revenue", "sales", "amount", "turnover", "ca", "chiffre", "income", "price", "mad", "usd"]
_UNITS_HINTS   = ["units", "qty", "quantity", "sold", "volume", "count"]
_PRODUCT_HINTS = ["category", "cat", "product", "item", "dept", "department", "rayon", "famille", "type", "group"]
_STORE_HINTS   = ["store", "branch", "location", "site", "magasin", "boutique", "shop"]
_PROMO_HINTS   = ["promo", "promotion", "discount", "sale", "offer", "markdown", "campaign"]


def _score_col(col: str, hints: list[str]) -> int:
    col_lower = col.lower().replace("_", " ").replace("-", " ")
    return sum(h in col_lower for h in hints)


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect the role of each column.
    Returns mapping: {role: column_name} for date, sales, product, store.
    """
    result = {}
    cols = list(df.columns)

    # Date: first try to parse, prefer hint-matching cols
    date_candidates = sorted(cols, key=lambda c: _score_col(c, _DATE_HINTS), reverse=True)
    for c in date_candidates:
        try:
            pd.to_datetime(df[c].dropna().iloc[:5])
            result["date"] = c
            break
        except Exception:
            continue

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Sales (revenue) — prefer revenue-like cols with larger values
    revenue_scores = [(c, _score_col(c, _SALES_HINTS), df[c].median()) for c in num_cols]
    revenue_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if revenue_scores:
        result["sales"] = revenue_scores[0][0]

    # Units sold — separate from revenue
    unit_scores = [(c, _score_col(c, _UNITS_HINTS)) for c in num_cols if c != result.get("sales")]
    unit_scores.sort(key=lambda x: x[1], reverse=True)
    if unit_scores and unit_scores[0][1] > 0:
        result["units"] = unit_scores[0][0]

    # Product/category — categorical col
    prod_scores = [(c, _score_col(c, _PRODUCT_HINTS)) for c in cat_cols]
    prod_scores.sort(key=lambda x: x[1], reverse=True)
    if prod_scores:
        result["product"] = prod_scores[0][0]

    # Store — categorical col, different from product
    store_scores = [(c, _score_col(c, _STORE_HINTS)) for c in cat_cols if c != result.get("product")]
    store_scores.sort(key=lambda x: x[1], reverse=True)
    if store_scores:
        result["store"] = store_scores[0][0]

    # Promotion
    promo_scores = [(c, _score_col(c, _PROMO_HINTS)) for c in df.columns]
    promo_scores.sort(key=lambda x: x[1], reverse=True)
    if promo_scores and promo_scores[0][1] > 0:
        result["promo"] = promo_scores[0][0]

    return result


def standardize(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Standardize any grocery CSV to the internal format:
    [date, store_id, category, sales, units_sold, is_promoted]
    """
    out = pd.DataFrame()

    if "date" in col_map:
        out["date"] = pd.to_datetime(df[col_map["date"]], errors="coerce")
    if "store" in col_map:
        out["store_id"] = df[col_map["store"]].astype(str)
    else:
        out["store_id"] = "All Stores"
    if "product" in col_map:
        out["category"] = df[col_map["product"]].astype(str)
    else:
        out["category"] = "All Categories"
    if "sales" in col_map:
        out["sales"] = pd.to_numeric(df[col_map["sales"]], errors="coerce").fillna(0)
    else:
        out["sales"] = 0.0
    if "units" in col_map:
        out["units_sold"] = pd.to_numeric(df[col_map["units"]], errors="coerce").fillna(0)
    else:
        out["units_sold"] = out["sales"] / 8.0  # estimate
    if "promo" in col_map:
        out["is_promoted"] = pd.to_numeric(df[col_map["promo"]], errors="coerce").fillna(0).astype(int)
    else:
        out["is_promoted"] = 0

    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal + lag + rolling features from standardized dataframe.
    Groups by (store_id, category).
    """
    df = df.copy().sort_values(["store_id", "category", "date"]).reset_index(drop=True)
    df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
    df["month"]          = df["date"].dt.month
    df["quarter"]        = df["date"].dt.quarter
    df["year"]           = df["date"].dt.year
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    grp = df.groupby(["store_id", "category"])["sales"]
    df["lag_1"]  = grp.shift(1)
    df["lag_2"]  = grp.shift(2)
    df["lag_4"]  = grp.shift(4)
    df["lag_8"]  = grp.shift(8)
    df["roll_mean_4"]  = grp.transform(lambda x: x.shift(1).rolling(4,  min_periods=1).mean())
    df["roll_mean_12"] = grp.transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    df["roll_std_4"]   = grp.transform(lambda x: x.shift(1).rolling(4,  min_periods=1).std().fillna(0))

    # Encode store and category
    df["store_code"]    = df["store_id"].astype("category").cat.codes
    df["category_code"] = df["category"].astype("category").cat.codes

    df = df.dropna(subset=["lag_8"]).reset_index(drop=True)
    return df


FEATURE_COLS = [
    "week_of_year", "month", "quarter", "year", "is_month_start", "is_month_end",
    "is_promoted", "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_mean_4", "roll_mean_12", "roll_std_4",
    "store_code", "category_code",
]
