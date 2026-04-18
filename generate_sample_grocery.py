"""Generate realistic grocery store CSV for testing KhazinaSmart."""
import pandas as pd
import numpy as np
import os

np.random.seed(2026)

STORES = ["Store_Casablanca", "Store_Rabat", "Store_Tanger"]
CATEGORIES = [
    "Fruits & Vegetables", "Bakery", "Dairy", "Meat & Seafood",
    "Beverages", "Snacks & Confectionery", "Frozen Foods", "Personal Care",
]
dates = pd.date_range("2022-01-03", "2024-01-01", freq="W-MON")

base_revenue = {
    "Fruits & Vegetables": 12000, "Bakery": 9500,  "Dairy": 11000,
    "Meat & Seafood": 18000,       "Beverages": 8000,  "Snacks & Confectionery": 6500,
    "Frozen Foods": 7000,          "Personal Care": 5000,
}
store_mult = {"Store_Casablanca": 1.4, "Store_Rabat": 1.0, "Store_Tanger": 0.75}

holiday_dates = set(pd.to_datetime([
    "2022-01-10", "2022-04-25", "2022-05-02", "2022-07-04", "2022-11-28", "2022-12-26",
    "2023-01-02", "2023-04-24", "2023-07-03", "2023-11-27", "2023-12-25",
]).strftime("%Y-%m-%d"))

rows = []
for store in STORES:
    for cat in CATEGORIES:
        base = base_revenue[cat] * store_mult[store]
        for date in dates:
            month = date.month
            seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (month - 3) / 12)
            if month in [11, 12]:
                seasonal += 0.35
            if month in [6, 7, 8] and cat == "Beverages":
                seasonal += 0.4
            if month in [3, 4, 5] and cat == "Fruits & Vegetables":
                seasonal += 0.3
            is_holiday = date.strftime("%Y-%m-%d") in holiday_dates
            holiday_bump = 1.25 if is_holiday else 1.0
            is_promo = np.random.choice([0, 1], p=[0.75, 0.25])
            promo_bump = 1.15 if is_promo else 1.0
            noise = np.random.normal(1.0, 0.06)
            rev = base * seasonal * holiday_bump * promo_bump * noise
            units = rev / np.random.uniform(4, 12)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "store_id": store,
                "category": cat,
                "units_sold": max(1, int(units)),
                "revenue": round(max(0, rev), 2),
                "is_promoted": is_promo,
                "temperature_c": round(
                    15 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 2), 1
                ),
                "is_holiday": int(is_holiday),
            })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_grocery.csv", index=False)
print(f"Generated: data/sample_grocery.csv — {len(df):,} rows")
print(f"  Stores: {df['store_id'].nunique()} | Categories: {df['category'].nunique()} | Weeks: {df['date'].nunique()}")
print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
print(f"  Revenue range: {df['revenue'].min():,.0f} – {df['revenue'].max():,.0f}")
