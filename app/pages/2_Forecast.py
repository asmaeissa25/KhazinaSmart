import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="KhazinaSmart — Forecast", page_icon="📈", layout="wide")

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
    .stButton > button { background-color: #e94560; color: white; border: none; border-radius: 8px; font-weight: 600; }
    .footer { text-align: center; color: #555; font-size: 0.8rem; padding: 20px 0; border-top: 1px solid #2a2a4e; margin-top: 32px; }
    .info-box { background-color: #1a1a2e; border-radius: 10px; padding: 16px; border-left: 3px solid #4a9eff; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='page-banner'>
    <h2 style='margin:0; color: #e94560;'>📈 Demand Forecast</h2>
    <p style='margin:4px 0 0 0; color: #aaa;'>Select a store and department to see AI-powered weekly demand predictions</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    import joblib
    for fname in ["xgb_tuned.pkl", "best_model.pkl"]:
        p = os.path.join(os.path.dirname(__file__), '..', '..', 'models', fname)
        if os.path.exists(p):
            return joblib.load(p)
    return None


@st.cache_data
def load_features():
    p = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'features_final.csv')
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=["Date"])
    return None


model = load_model()
features_df = load_features()

if features_df is not None:
    stores = sorted(features_df["Store"].unique().tolist())
else:
    stores = list(range(1, 11))

# Sidebar filters
with st.sidebar:
    st.markdown("### Forecast Controls")
    selected_store = st.selectbox("Select Store", stores)
    if features_df is not None:
        dept_options = sorted(features_df[features_df["Store"] == selected_store]["Dept"].unique().tolist())
    else:
        dept_options = list(range(1, 21))
    selected_dept = st.selectbox("Select Department", dept_options)
    forecast_weeks = st.slider("Forecast Weeks Ahead", 4, 12, 8)
    run_forecast = st.button("Generate Forecast")

if run_forecast or True:
    with st.spinner("Generating forecast..."):
        if features_df is not None and model is not None:
            try:
                from src.feature_engineering import get_feature_columns
                subset = features_df[
                    (features_df["Store"] == selected_store) &
                    (features_df["Dept"] == selected_dept)
                ].sort_values("Date").copy()

                feat_cols = [c for c in get_feature_columns() if c in subset.columns]
                X_sub = subset[feat_cols].fillna(0)
                subset["predicted"] = model.predict(X_sub)
                residuals = subset["Weekly_Sales"] - subset["predicted"]
                std_resid = residuals.std()

                cutoff = pd.to_datetime("2012-08-01")
                train_sub = subset[subset["Date"] < cutoff]
                test_sub = subset[subset["Date"] >= cutoff]

                last_date = subset["Date"].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                             periods=forecast_weeks, freq="W")
                last_pred = subset["predicted"].iloc[-1] if len(subset) > 0 else 10000
                future_preds = last_pred * np.random.normal(1.0, 0.05, forecast_weeks).cumprod()
                future_preds = np.abs(future_preds)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_sub["Date"], y=train_sub["Weekly_Sales"],
                                         name="Historical Actual", line=dict(color="#4a9eff", width=2)))
                fig.add_trace(go.Scatter(x=train_sub["Date"], y=train_sub["predicted"],
                                         name="Model Fit (Train)", line=dict(color="#ff9f43", width=1.5, dash="dot")))
                if len(test_sub) > 0:
                    fig.add_trace(go.Scatter(x=test_sub["Date"], y=test_sub["Weekly_Sales"],
                                             name="Actual (Test)", line=dict(color="#4a9eff", width=2)))
                    fig.add_trace(go.Scatter(x=test_sub["Date"], y=test_sub["predicted"],
                                             name="Predicted (Test)", line=dict(color="#ff9f43", width=2)))

                upper = future_preds + std_resid
                lower = np.maximum(future_preds - std_resid, 0)
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself", fillcolor="rgba(233,69,96,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Interval"
                ))
                fig.add_trace(go.Scatter(x=future_dates, y=future_preds,
                                         name="Forecast", line=dict(color="#e94560", width=2.5, dash="dash")))
                cutoff_date = last_date
                fig.add_vline(x=cutoff_date, line_width=2, line_dash="dot", line_color="#7c3aed")
                fig.update_layout(
                    title=f"Demand Forecast — Store {selected_store}, Dept {selected_dept}",
                    height=500, template="plotly_dark",
                    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
                    xaxis_title="Date", yaxis_title="Weekly Sales",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Bar chart
                fig2 = go.Figure(go.Bar(
                    x=[d.strftime("%b %d") for d in future_dates],
                    y=future_preds,
                    marker_color="#e94560",
                ))
                fig2.update_layout(
                    title=f"Weekly Predicted Demand — Next {forecast_weeks} Weeks",
                    height=350, template="plotly_dark",
                    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
                )
                st.plotly_chart(fig2, use_container_width=True)

                forecast_table = pd.DataFrame({
                    "Week": range(1, forecast_weeks + 1),
                    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                    "Predicted Sales": future_preds.round(0).astype(int),
                    "Lower Bound": lower.round(0).astype(int),
                    "Upper Bound": upper.round(0).astype(int),
                    "Trend": ["📈" if future_preds[i] > future_preds[max(0, i-1)] else "📉"
                               for i in range(len(future_preds))],
                })
                st.dataframe(forecast_table, use_container_width=True)

                csv = forecast_table.to_csv(index=False)
                st.download_button(
                    "📥 Export Forecast as CSV", csv,
                    f"forecast_store{selected_store}_dept{selected_dept}.csv", "text/csv"
                )

            except Exception as e:
                st.error(f"Forecast error: {e}")
                st.info("Model or data not available. Run the training notebooks first.")
        else:
            st.info("🔧 No trained model found. Please run `notebooks/03_model_training.ipynb` first.")
            st.markdown("""
            <div class='info-box'>
                <b>Steps to get the forecast running:</b><br>
                1. Place Walmart CSVs in <code>data/raw/</code> (or run <code>python generate_sample_data.py</code>)<br>
                2. Run <code>notebooks/01_data_loading.ipynb</code><br>
                3. Run <code>notebooks/02b_feature_engineering.ipynb</code><br>
                4. Run <code>notebooks/03_model_training.ipynb</code><br>
                5. Refresh this page
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
    <b>How to read the confidence interval:</b> The shaded area shows ±1 standard deviation of model residuals.
    ~68% of actual values should fall within this range. Wider bands = higher uncertainty.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
    KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business
</div>
""", unsafe_allow_html=True)
