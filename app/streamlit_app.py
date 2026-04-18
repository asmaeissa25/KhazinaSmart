import streamlit as st

st.set_page_config(
    page_title="KhazinaSmart",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global dark background */
    .stApp { background-color: #0f0f1a; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #16213e; }
    [data-testid="stSidebar"] * { color: #cccccc !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1a2e;
        border-left: 4px solid #e94560;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.15);
    }

    /* Headers */
    h1, h2, h3 { color: #ffffff !important; }

    /* Buttons */
    .stButton > button {
        background-color: #e94560;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #c73652; }

    /* Dataframe */
    [data-testid="stDataFrame"] { background-color: #1a1a2e; border-radius: 10px; }

    /* Input boxes */
    .stSelectbox > div > div, .stSlider { background-color: #1a1a2e; }

    /* Feature cards */
    .feature-card {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
        border-left: 4px solid #e94560;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Hero */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e94560, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Step card */
    .step-card {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2a4e;
        transition: border-color 0.3s;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        padding: 20px 0;
        border-top: 1px solid #2a2a4e;
        margin-top: 40px;
    }

    /* Morocco badge */
    .morocco-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e94560, #c8102e);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #e94560; font-size: 1.8rem; margin: 0;'>📦 KhazinaSmart</h1>
        <p style='color: #888; font-size: 0.85rem; margin: 8px 0 0 0;'>AI-Powered Inventory Intelligence</p>
    </div>
    <hr style='border-color: #2a2a4e;'>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    st.page_link("streamlit_app.py", label="🏠 Home", icon=None)
    st.page_link("pages/1_Dashboard.py", label="📊 Dashboard", icon=None)
    st.page_link("pages/2_Forecast.py", label="📈 Forecast", icon=None)
    st.page_link("pages/3_Alerts.py", label="⚠️ Alerts", icon=None)
    st.page_link("pages/4_Chatbot.py", label="🤖 KhazBot", icon=None)

    st.markdown("<hr style='border-color: #2a2a4e;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 0.75rem; color: #555; padding: 10px 0;'>
        IT DAY'Z Hackathon 2026<br>ENSA Tanger
    </div>
    """, unsafe_allow_html=True)

st.toast("Welcome to KhazinaSmart! 📦", icon="🎉")

# Hero section
st.markdown("""
<div style='text-align: center; padding: 60px 0 40px 0;'>
    <div class='hero-title'>KhazinaSmart</div>
    <p style='font-size: 1.3rem; color: #cccccc; max-width: 600px; margin: 0 auto;'>
        AI-Powered Inventory Intelligence for Moroccan SMEs
    </p>
    <br>
    <span class='morocco-badge'>🇲🇦 Morocco 2030 — AI for Startups & Business</span>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class='feature-card'>
        <h2 style='font-size: 2rem; margin: 0;'>📈</h2>
        <h3 style='color: #e94560; margin: 8px 0;'>Predict Demand</h3>
        <p style='color: #aaa; margin: 0;'>XGBoost ML model forecasts weekly demand per store & department with &lt;15% MAPE accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <h2 style='font-size: 2rem; margin: 0;'>⚠️</h2>
        <h3 style='color: #e94560; margin: 8px 0;'>Detect Overstock</h3>
        <p style='color: #aaa; margin: 0;'>Automatically flags inventory 30%+ above predicted demand, preventing capital freeze and waste.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <h2 style='font-size: 2rem; margin: 0;'>🤖</h2>
        <h3 style='color: #e94560; margin: 8px 0;'>AI Chatbot</h3>
        <p style='color: #aaa; margin: 0;'>Ask KhazBot anything about your inventory in French or English — powered by Claude AI.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# How it works
st.markdown("<h2 style='text-align: center; color: #fff;'>How It Works</h2>", unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
steps = [
    ("1️⃣", "Load Data", "Upload your store sales data or use integrated Walmart demo dataset"),
    ("2️⃣", "AI Analysis", "XGBoost model analyzes patterns, seasonality, and promotions"),
    ("3️⃣", "Risk Alerts", "Overstock and stockout risks are flagged with financial impact estimates"),
    ("4️⃣", "Take Action", "KhazBot recommends purchase orders and risk mitigation strategies"),
]
for col, (icon, title, desc) in zip([s1, s2, s3, s4], steps):
    with col:
        st.markdown(f"""
        <div class='step-card'>
            <div style='font-size: 2rem;'>{icon}</div>
            <h4 style='color: #e94560; margin: 8px 0;'>{title}</h4>
            <p style='color: #aaa; font-size: 0.85rem; margin: 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Stats row
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stores Monitored", "45", "Walmart dataset")
m2.metric("Data Points", "420K+", "Weekly sales records")
m3.metric("Model Accuracy", "R² > 0.85", "XGBoost")
m4.metric("Risk Detection", "Real-time", "Auto-classified")

st.markdown("""
<div class='footer'>
    KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business
</div>
""", unsafe_allow_html=True)
