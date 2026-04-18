"""
Feature engineering pipeline for KhazinaSmart.
Builds temporal, lag, rolling, promotional, and store features from the clean Walmart dataset.
"""
import pandas as pd
import numpy as np
import os

FEATURE_COLUMNS = [
    "week_of_year", "month", "quarter", "year", "is_month_start", "is_month_end",
    "IsHoliday",
    "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30", "rolling_std_30",
    "total_markdown", "has_markdown",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "store_type_A", "store_type_B", "store_type_C",
    "Size",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from the merged, cleaned dataframe.
    Returns the feature-enriched dataframe with NaN lag rows dropped.
    """
    df = df.copy()

    print(f"[FE] Step 1 — Sorting by Store, Dept, Date | shape: {df.shape}")
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    print(f"[FE] Step 2 — Temporal features | shape: {df.shape}")
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    print(f"[FE] Step 3 — Lag features (Store+Dept grouped) | shape: {df.shape}")
    grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]
    df["sales_lag_1"] = grp.shift(1)
    df["sales_lag_2"] = grp.shift(2)
    df["sales_lag_4"] = grp.shift(4)
    df["sales_lag_8"] = grp.shift(8)

    print(f"[FE] Step 4 — Rolling stats (Store+Dept grouped) | shape: {df.shape}")
    df["rolling_mean_7"] = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df["rolling_std_7"] = grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
    df["rolling_mean_30"] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df["rolling_std_30"] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))

    print(f"[FE] Step 5 — Promotional features | shape: {df.shape}")
    md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in md_cols:
        if col not in df.columns:
            df[col] = 0
    df["total_markdown"] = df[md_cols].sum(axis=1)
    df["has_markdown"] = (df["total_markdown"] > 0).astype(int)

    print(f"[FE] Step 6 — Store type encoding | shape: {df.shape}")
    if "Type" in df.columns:
        type_dummies = pd.get_dummies(df["Type"], prefix="store_type")
        for col in ["store_type_A", "store_type_B", "store_type_C"]:
            if col not in type_dummies.columns:
                type_dummies[col] = 0
        df = pd.concat([df, type_dummies[["store_type_A", "store_type_B", "store_type_C"]]], axis=1)
    else:
        df["store_type_A"] = 0
        df["store_type_B"] = 0
        df["store_type_C"] = 0

    print(f"[FE] Step 7 — Dropping NaN lag rows | shape before: {df.shape}")
    df = df.dropna(subset=["sales_lag_8"]).reset_index(drop=True)
    print(f"[FE] Step 7 done — shape after: {df.shape}")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features_final.csv", index=False)
    print(f"[FE] Saved features_final.csv — {df.shape[0]} rows, {df.shape[1]} cols")

    return df


def get_feature_columns() -> list:
    """Return list of feature column names for model input (X), excluding target and identifiers."""
    return [c for c in FEATURE_COLUMNS if c not in ("Weekly_Sales", "Date", "Store", "Dept")]


def get_train_test_split(df: pd.DataFrame, cutoff_date: str = "2012-08-01"):
    """
    Time-based train/test split on cutoff_date.
    Returns X_train, X_test, y_train, y_test.
    """
    features = get_feature_columns()
    available = [f for f in features if f in df.columns]

    cutoff = pd.to_datetime(cutoff_date)
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]

    X_train = train[available].fillna(0)
    X_test = test[available].fillna(0)
    y_train = train["Weekly_Sales"]
    y_test = test["Weekly_Sales"]

    print(f"[Split] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test
