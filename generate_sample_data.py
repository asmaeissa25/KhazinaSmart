"""Generate synthetic Walmart-like data for demo purposes when Kaggle data is unavailable."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

STORES = 10
DEPTS = 20
WEEKS = 145
START_DATE = "2010-02-05"

store_types = {i: "A" if i <= 4 else ("B" if i <= 7 else "C") for i in range(1, STORES + 1)}
store_sizes = {"A": 150000, "B": 100000, "C": 50000}
type_multiplier = {"A": 1.4, "B": 1.0, "C": 0.6}

dates = pd.date_range(start=START_DATE, periods=WEEKS, freq="W-FRI")

holiday_weeks = {
    "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
    "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
    "2012-02-10", "2012-09-07", "2012-11-23",
}
holiday_set = set(pd.to_datetime(list(holiday_weeks)).strftime("%Y-%m-%d"))

rows = []
for store in range(1, STORES + 1):
    stype = store_types[store]
    base_mult = type_multiplier[stype]
    for dept in range(1, DEPTS + 1):
        dept_base = np.random.uniform(5000, 50000)
        for week_idx, date in enumerate(dates):
            month = date.month
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
            if month in [11, 12]:
                seasonal += 0.4
            is_holiday = 1 if date.strftime("%Y-%m-%d") in holiday_set else 0
            holiday_bump = 1.2 if is_holiday else 1.0
            noise = np.random.normal(1.0, 0.08)
            sales = dept_base * base_mult * seasonal * holiday_bump * noise
            sales = max(0, sales)
            rows.append({
                "Store": store, "Dept": dept, "Date": date.strftime("%Y-%m-%d"),
                "Weekly_Sales": round(sales, 2), "IsHoliday": bool(is_holiday)
            })

train_df = pd.DataFrame(rows)

stores_df = pd.DataFrame([
    {"Store": s, "Type": store_types[s], "Size": store_sizes[store_types[s]]}
    for s in range(1, STORES + 1)
])

feat_rows = []
for store in range(1, STORES + 1):
    for date in dates:
        is_holiday = 1 if date.strftime("%Y-%m-%d") in holiday_set else 0
        feat_rows.append({
            "Store": store, "Date": date.strftime("%Y-%m-%d"),
            "Temperature": round(np.random.uniform(40, 100), 2),
            "Fuel_Price": round(np.random.uniform(2.5, 4.0), 3),
            "MarkDown1": round(np.random.choice([0, 0, 0, np.random.uniform(1000, 10000)]), 2),
            "MarkDown2": round(np.random.choice([0, 0, 0, np.random.uniform(500, 5000)]), 2),
            "MarkDown3": round(np.random.choice([0, 0, 0, np.random.uniform(200, 3000)]), 2),
            "MarkDown4": round(np.random.choice([0, 0, 0, np.random.uniform(300, 4000)]), 2),
            "MarkDown5": round(np.random.choice([0, 0, 0, np.random.uniform(400, 6000)]), 2),
            "CPI": round(np.random.uniform(126, 228), 3),
            "Unemployment": round(np.random.uniform(5, 14), 3),
            "IsHoliday": bool(is_holiday)
        })
features_df = pd.DataFrame(feat_rows)

os.makedirs("data/raw", exist_ok=True)
train_df.to_csv("data/raw/train.csv", index=False)
stores_df.to_csv("data/raw/stores.csv", index=False)
features_df.to_csv("data/raw/features.csv", index=False)

print(f"Generated synthetic data:")
print(f"  train.csv   — {len(train_df):,} rows ({STORES} stores x {DEPTS} depts x {WEEKS} weeks)")
print(f"  stores.csv  — {len(stores_df)} rows")
print(f"  features.csv — {len(features_df):,} rows")
print("Saved to data/raw/")
