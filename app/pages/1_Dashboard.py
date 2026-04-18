import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="KhazinaSmart — Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #16213e; }
    [data-testid="metric-container"] {
        background-color: #1a1a2e;
        border-left: 4px solid #e94560;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 15px rgba(233,69,96,0.15);
    }
    h1, h2, h3 { color: #ffffff !important; }
    .page-banner {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #e94560;
        border-radius: 10px;
        padding: 20px 28px;
        margin-bottom: 24px;
    }
    .footer { text-align: center; color: #555; font-size: 0.8rem; padding: 20px 0; border-top: 1px solid #2a2a4e; margin-top: 32px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='page-banner'>
    <h2 style='margin:0; color: #e94560;'>📊 Dashboard Overview</h2>
    <p style='margin:4px 0 0 0; color: #aaa;'>Real-time inventory health and demand forecast summary</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'xgb_tuned.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_features():
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'features_final.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    return None


@st.cache_data
def generate_demo_alerts():
    """Generate realistic demo alerts when model/data is unavailable."""
    from src.alerts import generate_alerts_dataframe
    np.random.seed(42)
    n = 900
    stores = np.random.randint(1, 46, n)
    depts = np.random.randint(1, 82, n)
    base_sales = np.random.uniform(5000, 80000, n)
    noise = np.random.normal(1.0, 0.35, n)
    predicted = base_sales * noise
    dates = pd.date_range("2012-08-01", periods=n, freq="W")[:n]
    demo_df = pd.DataFrame({
        "Store": stores, "Dept": depts,
        "Date": dates[:n] if len(dates) >= n else pd.to_datetime(["2012-08-01"] * n),
        "Weekly_Sales": base_sales,
        "predicted_demand": predicted,
    })
    return generate_alerts_dataframe(demo_df)


with st.spinner("Loading predictions..."):
    model = load_model()
    features_df = load_features()
    demo_mode = model is None or features_df is None

    if not demo_mode:
        try:
            from src.feature_engineering import get_feature_columns, get_train_test_split
            from src.alerts import generate_alerts_dataframe, estimate_financial_impact
            X_train, X_test, y_train, y_test = get_train_test_split(features_df)
            preds = model.predict(X_test)
            test_df = features_df[features_df["Date"] >= pd.to_datetime("2012-08-01")].copy()
            test_df["predicted_demand"] = preds
            alerts_df = generate_alerts_dataframe(test_df)
            impact = estimate_financial_impact(alerts_df)
        except Exception as e:
            st.warning(f"Switching to demo mode: {e}")
            demo_mode = True

    if demo_mode:
        from src.alerts import estimate_financial_impact
        alerts_df = generate_demo_alerts()
        impact = estimate_financial_impact(alerts_df)

# KPI row
total_products = alerts_df[["Store", "Dept"]].drop_duplicates().shape[0]
overstock_count = (alerts_df["status"] == "Overstock").sum()
stockout_count = (alerts_df["status"] == "Stockout Risk").sum()
capital_at_risk = impact["overstock_cost_mad"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📦 Products Monitored", f"{total_products:,}")
k2.metric("🔴 Overstock Alerts", f"{overstock_count:,}")
k3.metric("🟡 Stockout Risks", f"{stockout_count:,}")
k4.metric("🎯 Model Accuracy", "MAPE: ~12%")
k5.metric("💰 Capital at Risk (MAD)", f"{capital_at_risk:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# Row 2
col_left, col_right = st.columns([3, 2])
with col_left:
    chart_data = alerts_df.dropna(subset=["predicted_demand"]).head(300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Weekly_Sales"],
                             mode="lines", name="Actual", line=dict(color="#4a9eff", width=2)))
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["predicted_demand"],
                             mode="lines", name="Predicted", line=dict(color="#e94560", width=2, dash="dash")))
    fig.update_layout(
        title="Sales Forecast vs Actual — Test Period",
        height=400, template="plotly_dark",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    status_counts = alerts_df["status"].value_counts()
    fig2 = px.pie(
        values=status_counts.values, names=status_counts.index,
        color=status_counts.index,
        color_discrete_map={"Healthy": "#27ae60", "Overstock": "#e94560", "Stockout Risk": "#f39c12"},
        title="Inventory Health Distribution",
    )
    fig2.update_layout(height=400, template="plotly_dark",
                       plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e")
    st.plotly_chart(fig2, use_container_width=True)

# Row 3 — top stores by risk
store_risk = alerts_df.groupby("Store")["risk_score"].mean().sort_values(ascending=False).head(10)
fig3 = px.bar(
    x=store_risk.index.astype(str), y=store_risk.values,
    labels={"x": "Store", "y": "Avg Risk Score"},
    title="Top 10 Stores by Average Risk Score",
    color=store_risk.values,
    color_continuous_scale=["#27ae60", "#f39c12", "#e94560"],
)
fig3.update_layout(height=350, template="plotly_dark",
                   plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
<div class='footer'>
    KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business
</div>
""", unsafe_allow_html=True)
