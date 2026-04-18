# 🧠 KHAZINASMART — MASTER CONTEXT FOR CLAUDE
> Paste this ENTIRE file at the start of every new Claude session during the hackathon.
> Claude will have full context and will produce senior-level, production-ready code.

---

## 🎯 WHO WE ARE

We are a team of 3 Master's students in **"Intelligence Artificielle pour l'Économie Numérique et la Gestion"** (M2, 4th semester, internship phase) at Université Abdelmalek Essaâdi, Morocco.

We are participating in the **IT DAY'Z 11th Edition Hackathon** at ENSA Tanger, April 18-19, 2026.  
Theme: **AI for Startups & Business**  
Duration: **48 hours**  
Team size: **3 members**

---

## 🏆 THE PROJECT: KhazinaSmart

**KhazinaSmart** (Arabic: خزينة = treasury/stock + Smart) is an **AI-powered inventory prediction web platform** that helps startups, SMEs, and retail businesses:
- **Predict demand** for the coming weeks
- **Detect overstock** before products expire or freeze capital
- **Detect stockout risk** before sales are lost
- **Generate purchase recommendations** automatically
- **Answer natural language questions** about inventory via an AI chatbot (KhazBot)

---

## 📊 DATASET

**Walmart Store Sales Forecasting** (Kaggle public dataset)

### Files:
| File | Columns | Description |
|------|---------|-------------|
| `train.csv` | Store, Dept, Date, Weekly_Sales, IsHoliday | Main sales data |
| `stores.csv` | Store, Type (A/B/C), Size | Store metadata |
| `features.csv` | Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday | External features |

### Key Stats:
- 45 stores, ~81 departments each
- Weekly data: 2010-02-05 to 2012-11-01 (143 weeks)
- ~420,000+ rows
- Target variable: `Weekly_Sales`

### Why Walmart data:
Simulates real Moroccan SME challenges — multi-store, seasonal demand, promotional markdowns, holiday effects. In production, KhazinaSmart would connect to local ERP/POS systems via CSV upload or API.

---

## 🔧 TECH STACK

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Data | Pandas, NumPy |
| ML | **XGBoost** (primary), LightGBM (comparison), Scikit-learn |
| Time Series Baseline | Prophet (optional) |
| Visualization | Plotly (interactive), Seaborn, Matplotlib |
| Frontend | **Streamlit** (multi-page) |
| Backend | FastAPI (optional, for serving model) |
| Chatbot | Claude API or Gemini API + RAG over inventory dataframe |
| Notebooks | Jupyter in VS Code |
| Model Persistence | joblib (.pkl files) |
| Version Control | Git + GitHub |

---

## 📁 PROJECT STRUCTURE

```
KhazinaSmart/
├── data/
│   ├── raw/                    # Original Walmart CSVs
│   └── processed/              # Cleaned + feature-engineered data
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_overstock_detection.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── predict.py
│   ├── alerts.py
│   └── chatbot.py
├── app/
│   ├── streamlit_app.py        # Main entry point
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Forecast.py
│       ├── 3_Alerts.py
│       └── 4_Chatbot.py
├── models/
│   ├── best_model.pkl
│   └── xgb_tuned.pkl
├── reports/
└── requirements.txt
```

---

## 🤖 ML PIPELINE DECISIONS (IMPORTANT)

### Model Choice: XGBoost (NOT LSTM)
**We deliberately chose XGBoost over LSTM** because:
1. Walmart data is **structured tabular data** — tree models are superior here
2. LSTM needs GPU, 3D tensor reshaping, slow training — not viable in 48h
3. XGBoost handles mixed features (numerical + categorical) natively
4. Feature importance is interpretable — better for jury explanation
5. Industry benchmark: XGBoost consistently beats LSTM on tabular retail data

### Train/Test Split: TIME-BASED (NOT random)
- Train: all data before 2012-08-01
- Test: 2012-08-01 onwards
- Use `TimeSeriesSplit` for cross-validation
- **Never use random split on time series — it causes data leakage**

### Features to Engineer:
```python
# Temporal
week_of_year, month, quarter, year, is_holiday, is_month_start, is_month_end

# Lag features (grouped by Store + Dept)
sales_lag_1, sales_lag_2, sales_lag_4, sales_lag_8

# Rolling statistics (grouped by Store + Dept)
rolling_mean_7, rolling_std_7, rolling_mean_30, rolling_std_30

# External
temperature, fuel_price, CPI, unemployment

# Promotional
markdown_1..5 (fill NaN with 0), has_markdown (binary), total_markdown

# Store
store_type (A/B/C → one-hot), store_size
```

### Overstock / Stockout Logic:
```python
# After prediction:
predicted_demand = model.predict(X_test)
current_stock = Weekly_Sales (proxy, or user-input)

# Risk classification:
if current_stock > predicted_demand * 1.3:  → "Overstock" (30% above demand)
if current_stock < predicted_demand * 0.7:  → "Stockout Risk" (30% below demand)
else:                                        → "Healthy"
```

### Target Metrics:
| Metric | Target |
|--------|--------|
| RMSE | As low as possible |
| MAE | < 2000 (weekly sales units) |
| MAPE | < 15% |
| R² | > 0.85 |

---

## 🎨 DASHBOARD SPECS (Streamlit)

### Page 1 — Overview
- 5 KPI cards: Total Products, Overstock Alerts, Stockout Risks, Forecast Accuracy (%), Estimated Savings (MAD)
- Line chart: overall weekly sales trend (actual vs predicted)
- Pie chart: Healthy / Overstock / Stockout distribution

### Page 2 — Forecast
- Sidebar filters: Store selector, Department selector, Weeks ahead (4/8/12)
- Main chart: Plotly line chart — actual (blue) vs predicted (orange) with shaded confidence interval
- Table: Next N weeks predictions with confidence bounds

### Page 3 — Alerts
- Sortable, filterable table: Store | Dept | Product | Status | Current | Predicted | Risk Score | Action
- Color coding: 🔴 Overstock / 🟡 Stockout Risk / 🟢 Healthy
- Export to CSV button

### Page 4 — KhazBot Chatbot
- `st.chat_message` + `st.chat_input`
- 5 starter question buttons
- Responses grounded in actual inventory data

### UI Requirements:
- `st.set_page_config(layout="wide", page_title="KhazinaSmart")`
- Dark professional theme
- KhazinaSmart logo/brand in sidebar
- Plotly charts (NOT matplotlib) for all interactive visuals
- `@st.cache_data` and `@st.cache_resource` for performance

---

## 💬 CHATBOT SPECS (KhazBot)

### System Prompt:
```
You are KhazBot, the AI inventory assistant for KhazinaSmart.
You help business owners understand their stock health and make smart purchasing decisions.
You have access to real inventory prediction data.
Always reference specific numbers, product names, stores, or departments in your answers.
Be concise, actionable, and professional.
Answer in the same language the user writes in (French or English).
```

### Example queries to handle:
- "Which products are most overstocked?"
- "What should I order this week?"
- "Show me the budget needed for department 7"
- "What is my stockout risk for store 4?"
- "Which store is performing best?"
- "What are the seasonal trends for the next month?"

### Architecture:
1. User types question → parse intent
2. Query the predictions dataframe → extract relevant rows
3. Format context as string → send to LLM API with system prompt
4. Return natural language answer

---

## 🎯 WHAT THE JURY EVALUATES

1. **Working demo** (most important — must run live)
2. **Technical depth** (ML model quality, feature engineering, architecture)
3. **Business impact** (Morocco 2030 alignment, SME problem solved)
4. **UI quality** (professional, intuitive, visual)
5. **Pitch clarity** (problem → solution → demo → impact in 5 min)

---

## 🇲🇦 MOROCCO 2030 ALIGNMENT (for pitch)

| Morocco 2030 Pillar | KhazinaSmart |
|--------------------|-------------|
| SME Competitiveness | AI inventory for the 95% SME economic fabric |
| AI Made in Morocco | Moroccan students building sovereign AI solutions |
| 3,000 Startups by 2030 | Reduces #1 startup killer: cash & inventory waste |
| Digital Inclusion | Simple web platform, no technical skills needed |

---

## ⚠️ CRITICAL RULES FOR CLAUDE DURING HACKATHON

1. **Always use XGBoost as primary model** — not LSTM, not neural networks
2. **Always time-based train/test split** — never random split on time series
3. **Fill MarkDown NaN with 0** — not mean/median (NaN means no promotion)
4. **Group lag/rolling features by (Store, Dept)** — never compute globally
5. **Use Plotly for all Streamlit charts** — not matplotlib
6. **@st.cache_data on all data loading functions** — performance critical
7. **Every Jupyter cell must have a markdown header** — jury will scroll notebooks
8. **All charts must have titles, axis labels, and legends** — visual quality matters
9. **Handle edge cases gracefully** — try/except on all API calls
10. **Target: working > perfect** — ship a demo that runs, not a perfect one that crashes
