import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="KhazinaSmart — Alerts", page_icon="⚠️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #16213e; }
    h1, h2, h3 { color: #ffffff !important; }
    .page-banner {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #e94560;
        border-radius: 10px;
        padding: 20px 28px;
        margin-bottom: 24px;
    }
    [data-testid="metric-container"] {
        background-color: #1a1a2e;
        border-left: 4px solid #e94560;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 15px rgba(233,69,96,0.15);
    }
    .stButton > button { background-color: #e94560; color: white; border: none; border-radius: 8px; font-weight: 600; }
    .footer { text-align: center; color: #555; font-size: 0.8rem; padding: 20px 0; border-top: 1px solid #2a2a4e; margin-top: 32px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='page-banner'>
    <h2 style='margin:0; color: #e94560;'>⚠️ Inventory Risk Alerts</h2>
    <p style='margin:4px 0 0 0; color: #aaa;'>Monitor overstock and stockout risks across all stores and departments</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def get_alerts():
    from src.alerts import generate_alerts_dataframe, estimate_financial_impact
    np.random.seed(42)
    n = 900
    stores = np.random.randint(1, 46, n)
    depts = np.random.randint(1, 82, n)
    base = np.random.uniform(5000, 80000, n)
    noise = np.random.normal(1.0, 0.35, n)
    predicted = base * noise
    demo_df = pd.DataFrame({
        "Store": stores, "Dept": depts,
        "Date": pd.date_range("2012-08-01", periods=n, freq="W")[:n],
        "Weekly_Sales": base,
        "predicted_demand": predicted,
    })

    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'features_final.csv')
    model_path_tuned = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'xgb_tuned.pkl')
    model_path_best = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_model.pkl')

    try:
        import joblib
        from src.feature_engineering import get_feature_columns, get_train_test_split
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, parse_dates=["Date"])
            model_p = model_path_tuned if os.path.exists(model_path_tuned) else model_path_best
            if os.path.exists(model_p):
                model = joblib.load(model_p)
                X_train, X_test, y_train, y_test = get_train_test_split(df)
                preds = model.predict(X_test)
                test_df = df[df["Date"] >= pd.to_datetime("2012-08-01")].copy()
                test_df["predicted_demand"] = preds
                return generate_alerts_dataframe(test_df), estimate_financial_impact(generate_alerts_dataframe(test_df))
    except Exception:
        pass

    alerts = generate_alerts_dataframe(demo_df)
    return alerts, estimate_financial_impact(alerts)


alerts_df, impact = get_alerts()

# Filters row
f1, f2, f3, f4 = st.columns([2, 3, 2, 1])
with f1:
    status_filter = st.selectbox("Status", ["All", "Overstock", "Stockout Risk", "Healthy"])
with f2:
    store_options = sorted(alerts_df["Store"].unique().tolist())
    store_filter = st.multiselect("Stores", store_options, default=store_options[:5])
with f3:
    min_risk = st.slider("Min Risk Score", 0, 100, 0)
with f4:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh = st.button("Refresh")

# Apply filters
filtered = alerts_df.copy()
if status_filter != "All":
    filtered = filtered[filtered["status"] == status_filter]
if store_filter:
    filtered = filtered[filtered["Store"].isin(store_filter)]
filtered = filtered[filtered["risk_score"] >= min_risk]

# Summary metrics
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
critical = filtered[filtered["risk_score"] > 70]
warning = filtered[(filtered["risk_score"] >= 40) & (filtered["risk_score"] <= 70)]
healthy = filtered[filtered["status"] == "Healthy"]

m1.metric("🔴 Critical (risk > 70)",
          f"{len(critical):,}",
          f"{impact['overstock_cost_mad']:,.0f} MAD at risk")
m2.metric("🟡 Warning (risk 40–70)", f"{len(warning):,}")
m3.metric("🟢 Healthy", f"{len(healthy):,}")

st.markdown("<br>", unsafe_allow_html=True)

# Alerts table
display_cols = ["Store", "Dept", "status", "risk_score", "Weekly_Sales", "predicted_demand", "action_needed"]
display_cols = [c for c in display_cols if c in filtered.columns]
display_df = filtered[display_cols].head(200)

st.dataframe(
    display_df,
    column_config={
        "status": st.column_config.SelectboxColumn("Status", options=["Healthy", "Overstock", "Stockout Risk"]),
        "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
        "Weekly_Sales": st.column_config.NumberColumn("Current Sales", format="%.0f"),
        "predicted_demand": st.column_config.NumberColumn("Predicted Demand", format="%.0f"),
        "action_needed": st.column_config.TextColumn("Action Needed"),
    },
    use_container_width=True,
    height=400,
)

csv = filtered.to_csv(index=False)
st.download_button("📥 Export Alerts as CSV", csv, "khazinasmart_alerts.csv", "text/csv")

st.markdown("<br>", unsafe_allow_html=True)

# Risk distribution histogram
col_a, col_b = st.columns(2)
with col_a:
    fig1 = px.histogram(
        filtered, x="risk_score", color="status",
        color_discrete_map={"Healthy": "#27ae60", "Overstock": "#e94560", "Stockout Risk": "#f39c12"},
        nbins=30, title="Risk Score Distribution",
        labels={"risk_score": "Risk Score", "count": "Count"},
    )
    fig1.update_layout(height=350, template="plotly_dark",
                       plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e")
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    scatter_data = filtered.head(500)
    fig2 = px.scatter(
        scatter_data, x="Store", y="Dept",
        color="risk_score", size="risk_score",
        color_continuous_scale=["#27ae60", "#f39c12", "#e94560"],
        title="Store × Dept Risk Heatmap",
        labels={"risk_score": "Risk Score"},
        size_max=20,
    )
    fig2.update_layout(height=350, template="plotly_dark",
                       plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div style='text-align: center; color: #555; font-size: 0.8rem; padding: 20px 0; border-top: 1px solid #2a2a4e; margin-top: 32px;'>
    KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business
</div>
""", unsafe_allow_html=True)
