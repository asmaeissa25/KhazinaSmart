import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="KhazinaSmart — KhazBot", page_icon="🤖", layout="wide")

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
    .stButton > button {
        background-color: #1a1a2e;
        color: #e94560;
        border: 1px solid #e94560;
        border-radius: 20px;
        font-size: 0.85rem;
        padding: 6px 14px;
        margin: 4px;
    }
    .stButton > button:hover { background-color: #e94560; color: white; }
    .stChatMessage { background-color: #1a1a2e !important; border-radius: 10px !important; }
    .footer { text-align: center; color: #555; font-size: 0.8rem; padding: 20px 0; border-top: 1px solid #2a2a4e; margin-top: 32px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='page-banner'>
    <h2 style='margin:0; color: #e94560;'>🤖 KhazBot — Your AI Inventory Assistant</h2>
    <p style='margin:4px 0 0 0; color: #aaa;'>Ask anything about your inventory in French or English</p>
</div>
""", unsafe_allow_html=True)

from src.chatbot import get_starter_questions, answer_inventory_question
from src.alerts import generate_alerts_dataframe, estimate_financial_impact


@st.cache_data
def get_demo_alerts():
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
    return generate_alerts_dataframe(demo_df)


alerts_df = get_demo_alerts()
impact = estimate_financial_impact(alerts_df)

# Sidebar
with st.sidebar:
    st.markdown("### KhazBot Settings")

    api_key = st.text_input("Claude API Key (optional)", type="password",
                            help="Enter Anthropic API key for full AI responses")

    with st.expander("💡 Sample Questions"):
        for q in get_starter_questions():
            st.markdown(f"- {q}")

    with st.expander("📊 Data Summary"):
        st.markdown(f"""
        - **Total items:** {len(alerts_df):,}
        - **Overstock:** {(alerts_df['status']=='Overstock').sum():,}
        - **Stockout Risk:** {(alerts_df['status']=='Stockout Risk').sum():,}
        - **Healthy:** {(alerts_df['status']=='Healthy').sum():,}
        - **Capital at risk:** {impact['total_risk_mad']:,.0f} MAD
        """)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm **KhazBot**, your AI inventory assistant.\n\nI can analyze your inventory health, identify risks, and recommend actions. Ask me anything in **French or English**!"
    })

# Starter question buttons
st.markdown("**Quick Questions:**")
starter_cols = st.columns(5)
starters = get_starter_questions()
for i, (col, question) in enumerate(zip(starter_cols, starters)):
    with col:
        if st.button(question, key=f"starter_{i}"):
            st.session_state.pending_question = question

# Display chat history
for msg in st.session_state.messages:
    avatar = "📦" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle pending starter question
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📦"):
        with st.spinner("KhazBot is analyzing your inventory..."):
            key = api_key if api_key else None
            response = answer_inventory_question(question, alerts_df, api_key=key)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input
if prompt := st.chat_input("Ask KhazBot about your inventory..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="📦"):
        with st.spinner("KhazBot is analyzing your inventory..."):
            key = api_key if api_key else None
            response = answer_inventory_question(prompt, alerts_df, api_key=key)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("""
<div class='footer'>
    KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business
</div>
""", unsafe_allow_html=True)
