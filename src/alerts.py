"""
Inventory risk classification engine for KhazinaSmart.
Classifies inventory as Overstock, Stockout Risk, or Healthy based on predicted demand.
"""
import pandas as pd
import numpy as np


def classify_inventory_risk(current_stock: float, predicted_demand: float) -> str:
    """
    Classify inventory status based on current stock vs predicted demand.

    Returns 'Overstock', 'Stockout Risk', or 'Healthy'.
    """
    if predicted_demand <= 0:
        return "Healthy"
    if current_stock > predicted_demand * 1.3:
        return "Overstock"
    if current_stock < predicted_demand * 0.7:
        return "Stockout Risk"
    return "Healthy"


def compute_risk_score(current_stock: float, predicted_demand: float) -> float:
    """
    Compute a risk score from 0 (balanced) to 100 (critical).

    Formula: |current - predicted| / predicted * 100, capped at 100.
    """
    if predicted_demand <= 0:
        return 0.0
    score = abs(current_stock - predicted_demand) / predicted_demand * 100
    return min(score, 100.0)


def generate_alerts_dataframe(df_with_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Generate full alerts dataframe with status, risk score, and action recommendations.

    Input must have columns: Store, Dept, Date, Weekly_Sales, predicted_demand.
    Returns dataframe sorted by risk_score descending.
    """
    df = df_with_predictions.copy()
    df["status"] = df.apply(
        lambda r: classify_inventory_risk(r["Weekly_Sales"], r["predicted_demand"]), axis=1
    )
    df["risk_score"] = df.apply(
        lambda r: round(compute_risk_score(r["Weekly_Sales"], r["predicted_demand"]), 2), axis=1
    )

    def _action(row):
        if row["status"] == "Overstock":
            excess = int(row["Weekly_Sales"] - row["predicted_demand"])
            return f"Reduce orders — excess: {excess:,} units"
        if row["status"] == "Stockout Risk":
            deficit = int(row["predicted_demand"] - row["Weekly_Sales"])
            return f"Increase orders — deficit: {deficit:,} units"
        return "No action needed"

    df["action_needed"] = df.apply(_action, axis=1)
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def get_top_alerts(alerts_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top N highest-risk items from the alerts dataframe."""
    return alerts_df.head(n)


def estimate_financial_impact(alerts_df: pd.DataFrame, avg_unit_cost: float = 50.0) -> dict:
    """
    Estimate financial impact of overstock and stockout situations.

    Returns dict with overstock_cost_mad, stockout_lost_sales_mad, total_risk_mad, items_at_risk.
    """
    overstock = alerts_df[alerts_df["status"] == "Overstock"].copy()
    stockout = alerts_df[alerts_df["status"] == "Stockout Risk"].copy()

    overstock_cost = (
        (overstock["Weekly_Sales"] - overstock["predicted_demand"]).clip(lower=0) * avg_unit_cost
    ).sum()
    stockout_loss = (
        (stockout["predicted_demand"] - stockout["Weekly_Sales"]).clip(lower=0) * avg_unit_cost
    ).sum()

    return {
        "overstock_cost_mad": round(overstock_cost, 2),
        "stockout_lost_sales_mad": round(stockout_loss, 2),
        "total_risk_mad": round(overstock_cost + stockout_loss, 2),
        "items_at_risk": len(alerts_df[alerts_df["status"] != "Healthy"]),
    }
