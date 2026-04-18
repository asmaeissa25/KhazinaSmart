"""
KhazBot AI engine for KhazinaSmart.
Answers natural language questions about inventory using Claude API or rule-based fallback.
"""
import pandas as pd

KHAZBOT_SYSTEM_PROMPT = """You are KhazBot, the AI inventory assistant for KhazinaSmart — an intelligent inventory management platform.
You help business owners understand their stock health and make smart decisions.
You have access to real inventory data provided as context.
ALWAYS reference specific numbers, stores, departments, or products from the data.
Be concise, professional, and actionable.
Format numbers clearly (e.g., "1,250 units", "45,000 MAD").
Answer in the same language the user writes in (French or English).
Never say you don't have data — analyze what's provided and give your best answer."""


def format_inventory_context(alerts_df: pd.DataFrame) -> str:
    """
    Convert top 50 rows of alerts_df to a clean string summary for LLM context.
    """
    top = alerts_df.head(50)
    total = len(alerts_df)
    overstock_count = (alerts_df["status"] == "Overstock").sum()
    stockout_count = (alerts_df["status"] == "Stockout Risk").sum()
    healthy_count = (alerts_df["status"] == "Healthy").sum()

    top_overstock = alerts_df[alerts_df["status"] == "Overstock"].head(5)
    top_stockout = alerts_df[alerts_df["status"] == "Stockout Risk"].head(5)

    total_risk = (alerts_df["risk_score"].sum() * 50).round(0)

    lines = [
        f"=== INVENTORY SUMMARY ===",
        f"Total items monitored: {total:,}",
        f"Overstock: {overstock_count} | Stockout Risk: {stockout_count} | Healthy: {healthy_count}",
        f"Total capital at risk: {total_risk:,.0f} MAD",
        "",
        "TOP 5 OVERSTOCK ITEMS:",
    ]
    for _, row in top_overstock.iterrows():
        lines.append(
            f"  Store {row['Store']}, Dept {row['Dept']} — Risk: {row['risk_score']:.1f} — {row['action_needed']}"
        )

    lines.append("\nTOP 5 STOCKOUT RISK ITEMS:")
    for _, row in top_stockout.iterrows():
        lines.append(
            f"  Store {row['Store']}, Dept {row['Dept']} — Risk: {row['risk_score']:.1f} — {row['action_needed']}"
        )

    return "\n".join(lines)


def answer_inventory_question(question: str, alerts_df: pd.DataFrame, api_key: str = None) -> str:
    """
    Answer a natural language inventory question using Claude API or rule-based fallback.

    Returns a formatted string response.
    """
    context = format_inventory_context(alerts_df)

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            user_message = f"Inventory data:\n{context}\n\nQuestion: {question}"
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=KHAZBOT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as e:
            pass

    return _rule_based_answer(question, alerts_df, context)


def _rule_based_answer(question: str, alerts_df: pd.DataFrame, context: str) -> str:
    q = question.lower()
    overstock = alerts_df[alerts_df["status"] == "Overstock"]
    stockout = alerts_df[alerts_df["status"] == "Stockout Risk"]
    healthy = alerts_df[alerts_df["status"] == "Healthy"]

    if "overstock" in q or "surplus" in q or "trop" in q:
        if overstock.empty:
            return "✅ Great news! No overstock situations detected in your inventory."
        lines = [f"🔴 **{len(overstock)} overstock situations detected:**\n"]
        for _, row in overstock.head(5).iterrows():
            lines.append(f"- **Store {row['Store']}, Dept {row['Dept']}** — Risk score: {row['risk_score']:.1f} — {row['action_needed']}")
        return "\n".join(lines)

    if "order" in q or "acheter" in q or "commander" in q or "stock" in q:
        if stockout.empty:
            return "✅ No immediate stockout risks. Your inventory levels are healthy across all monitored items."
        lines = [f"📦 **{len(stockout)} items need restocking this week:**\n"]
        for _, row in stockout.head(5).iterrows():
            lines.append(f"- **Store {row['Store']}, Dept {row['Dept']}** — {row['action_needed']} — Risk: {row['risk_score']:.1f}")
        return "\n".join(lines)

    if "budget" in q or "cost" in q or "capital" in q or "mad" in q or "argent" in q:
        total_risk = (alerts_df["risk_score"].sum() * 50)
        overstock_cost = (overstock["risk_score"].sum() * 50)
        stockout_cost = (stockout["risk_score"].sum() * 50)
        return (
            f"💰 **Financial Risk Summary:**\n\n"
            f"- Total capital at risk: **{total_risk:,.0f} MAD**\n"
            f"- Overstock exposure: **{overstock_cost:,.0f} MAD** ({len(overstock)} items)\n"
            f"- Stockout lost sales: **{stockout_cost:,.0f} MAD** ({len(stockout)} items)\n"
            f"- Items at risk: **{len(overstock) + len(stockout)}** out of {len(alerts_df)}"
        )

    if "best" in q or "meilleur" in q or "top" in q or "store" in q:
        if "predicted_demand" in alerts_df.columns:
            store_sales = alerts_df.groupby("Store")["predicted_demand"].mean().sort_values(ascending=False)
            top_store = store_sales.index[0]
            return (
                f"🏪 **Best performing store: Store {top_store}**\n\n"
                f"Average predicted weekly sales: **{store_sales.iloc[0]:,.0f} units**\n\n"
                f"Top 3 stores by predicted demand:\n"
                + "\n".join([f"- Store {s}: {v:,.0f}" for s, v in store_sales.head(3).items()])
            )

    total = len(alerts_df)
    return (
        f"📊 **KhazinaSmart Inventory Health Summary:**\n\n"
        f"- Total items monitored: **{total:,}**\n"
        f"- 🔴 Overstock alerts: **{len(overstock)}** ({len(overstock)/total*100:.1f}%)\n"
        f"- 🟡 Stockout risks: **{len(stockout)}** ({len(stockout)/total*100:.1f}%)\n"
        f"- 🟢 Healthy items: **{len(healthy)}** ({len(healthy)/total*100:.1f}%)\n\n"
        f"Your top priority: address the **{len(overstock)+len(stockout)} at-risk items** to protect cash flow."
    )


def get_starter_questions() -> list:
    """Return list of suggested starter questions for KhazBot."""
    return [
        "Which products are most overstocked? 🔴",
        "What should I order this week? 📦",
        "What is the total budget at risk? 💰",
        "Which store has the highest risk? 🏪",
        "Show me a summary of inventory health 📊",
    ]
