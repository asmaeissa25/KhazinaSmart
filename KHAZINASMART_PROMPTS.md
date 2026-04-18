# ⚡ KHAZINASMART — CLAUDE EXECUTION PROMPTS
> Use these prompts in order during the hackathon.
> Always paste KHAZINASMART_CONTEXT.md FIRST in each new Claude session, then use the prompt below.

---

## 🔰 HOW TO USE THIS FILE

1. Open Claude in VS Code (Claude extension) or claude.ai
2. **Start every session with:** `"Read this context first: [paste full KHAZINASMART_CONTEXT.md]"`
3. Then use the prompt from the phase you are working on
4. Claude will produce production-ready, hackathon-quality code

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1 — SETUP & DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT 1.1 — Project Structure & Requirements

```
You have the full KhazinaSmart context above.

Create the complete project structure for KhazinaSmart:
1. Generate all folders and empty __init__.py files
2. Create requirements.txt with exact versions:
   pandas>=2.0, numpy>=1.24, scikit-learn>=1.3, xgboost>=2.0,
   lightgbm>=4.0, streamlit>=1.28, plotly>=5.17, joblib>=1.3,
   prophet>=1.1 (optional), fastapi>=0.104, uvicorn>=0.24
3. Create a .gitignore for Python projects
4. Create a README.md with: project description, team, setup instructions (pip install -r requirements.txt, streamlit run app/streamlit_app.py)
5. Create src/__init__.py, app/__init__.py

Output each file separately with its full path.
```

---

## PROMPT 1.2 — Data Loading & Cleaning Notebook

```
You have the full KhazinaSmart context above.

Create notebooks/01_data_loading.ipynb as a complete Jupyter notebook.

The notebook must:
- ## Cell 1 (markdown): "# KhazinaSmart — Data Loading & Cleaning" + description
- ## Cell 2: imports (pandas, numpy, os, warnings)
- ## Cell 3 (markdown): "## 1. Load Raw Data"
- ## Cell 4: Load train.csv, stores.csv, features.csv from data/raw/
- ## Cell 5 (markdown): "## 2. Initial Inspection"
- ## Cell 6: For each dataframe: .shape, .dtypes, .head(3), .isnull().sum()
- ## Cell 7 (markdown): "## 3. Merge DataFrames"
- ## Cell 8: Merge train + stores on 'Store', then merge with features on ['Store','Date','IsHoliday']. Convert Date to datetime.
- ## Cell 9 (markdown): "## 4. Handle Missing Values"
- ## Cell 10: Fill MarkDown1-5 NaN with 0 (no promotion = 0). Show null counts before/after.
- ## Cell 11 (markdown): "## 5. Data Quality Report"
- ## Cell 12: Print total rows, date range, number of stores, number of departments, % holiday weeks, min/max/mean Weekly_Sales
- ## Cell 13 (markdown): "## 6. Save Clean Data"
- ## Cell 14: Save to data/processed/walmart_clean.csv. Print "Saved: {rows} rows, {cols} columns"

Every code cell must have a corresponding markdown header cell.
Output as valid JSON notebook format.
```

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2 — EDA & FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT 2.1 — EDA Notebook (Full Visual Analysis)

```
You have the full KhazinaSmart context above.

Create notebooks/02_EDA.ipynb. Use Plotly for all charts (not matplotlib).
The notebook must produce these exact visualizations:

1. (markdown) "# KhazinaSmart — Exploratory Data Analysis"
2. Load data/processed/walmart_clean.csv
3. CHART 1 — Weekly Sales Distribution:
   - Plotly histogram of Weekly_Sales with color=#1a1a2e
   - Add vertical line at mean and median
   - Title: "Weekly Sales Distribution across all Stores & Departments"

4. CHART 2 — Sales Trend Over Time:
   - Aggregate Weekly_Sales by Date (sum)
   - Plotly line chart, color=#e94560
   - Separate traces for holiday vs non-holiday weeks
   - Title: "Total Weekly Sales Trend (2010-2012)"

5. CHART 3 — Sales by Store Type:
   - Grouped bar chart: Store Type (A/B/C) × avg Weekly_Sales
   - Title: "Average Weekly Sales by Store Type"

6. CHART 4 — Top 10 & Bottom 10 Departments:
   - Horizontal bar chart, green for top 10, red for bottom 10
   - Title: "Top 10 vs Bottom 10 Departments by Average Sales"

7. CHART 5 — Holiday Impact:
   - Box plot: IsHoliday=True vs False on Weekly_Sales
   - Title: "Impact of Holiday Weeks on Sales"

8. CHART 6 — Correlation Heatmap:
   - Select numeric columns, compute corr(), plot with Plotly imshow
   - Title: "Feature Correlation Heatmap"

9. CHART 7 — Monthly Seasonality:
   - Average sales per month (1-12), bar chart
   - Title: "Monthly Sales Seasonality"

10. CHART 8 — Markdown vs Sales:
    - Scatter plot: total_markdown (sum of MD1-5) vs Weekly_Sales
    - Color by IsHoliday
    - Title: "Promotional Markdowns vs Weekly Sales"

After each chart, add a markdown cell with 2-3 key insights.
All charts: height=450, template='plotly_white', show fig.show()
```

---

## PROMPT 2.2 — Feature Engineering Module

```
You have the full KhazinaSmart context above.

Create src/feature_engineering.py as a production-quality Python module.

Requirements:
- Module docstring explaining what it does
- Function: build_features(df: pd.DataFrame) -> pd.DataFrame
  Steps inside build_features():
  1. Sort by ['Store', 'Dept', 'Date'] — critical for lag features
  2. Temporal features:
     df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
     df['month'] = df['Date'].dt.month
     df['quarter'] = df['Date'].dt.quarter
     df['year'] = df['Date'].dt.year
     df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
     df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
  3. Lag features (grouped by Store + Dept):
     sales_lag_1, sales_lag_2, sales_lag_4, sales_lag_8
  4. Rolling stats (grouped by Store + Dept, min_periods=1):
     rolling_mean_7, rolling_std_7 (7-week window)
     rolling_mean_30, rolling_std_30 (30-week window)
  5. Promotional features:
     total_markdown = MarkDown1+...+MarkDown5 (already filled with 0)
     has_markdown = (total_markdown > 0).astype(int)
  6. Store type encoding: pd.get_dummies on Type column, prefix='store_type'
  7. Drop rows where lag features are NaN (first 8 rows per Store+Dept)
  8. Return final dataframe

- Function: get_feature_columns() -> list
  Returns list of all feature columns (X) to use for modeling
  Excludes: Weekly_Sales, Date, Store, Dept

- Function: get_train_test_split(df, cutoff_date='2012-08-01'):
  Time-based split. Returns X_train, X_test, y_train, y_test

Add logging at each step: print(f"[FE] Step X done — shape: {df.shape}")
Save final features: df.to_csv('data/processed/features_final.csv', index=False)
```

---

## PROMPT 2.3 — Feature Engineering Notebook

```
You have the full KhazinaSmart context above.

Create notebooks/02b_feature_engineering.ipynb that:
1. Imports and calls src/feature_engineering.py functions
2. Shows df.shape before and after each transformation step
3. Prints list of all generated features
4. Visualizes lag feature correlation with target (Weekly_Sales) — bar chart
5. Saves features_final.csv
6. Final cell prints: "Feature engineering complete. Total features: X. Total rows: Y"

Every cell has a markdown header.
```

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3 — MODEL TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT 3.1 — Model Training Notebook (Main)

```
You have the full KhazinaSmart context above.

Create notebooks/03_model_training.ipynb. This is the MOST IMPORTANT notebook — it must be clean, professional, and produce impressive outputs.

Structure:
1. (markdown) "# KhazinaSmart — Model Training & Evaluation"
2. Load features_final.csv
3. Call get_train_test_split() — show train/test sizes
4. (markdown) "## Baseline: Linear Regression"
5. Train LinearRegression, compute RMSE/MAE/MAPE/R2 on test set, print results
6. (markdown) "## Primary Model: XGBoost"
7. Train XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
8. Compute RMSE/MAE/MAPE/R2, print results
9. (markdown) "## Comparison: LightGBM"
10. Train LGBMRegressor with similar params, compute metrics
11. (markdown) "## Model Comparison Table"
12. Print a clean comparison DataFrame:
    | Model | RMSE | MAE | MAPE | R2 |
13. (markdown) "## Actual vs Predicted — XGBoost"
14. Plotly line chart: first 300 test samples, actual (blue) vs predicted (orange)
    Title: "XGBoost: Actual vs Predicted Weekly Sales"
15. (markdown) "## Feature Importance — Top 20"
16. Plotly horizontal bar chart of top 20 features by XGBoost importance
    Color: gradient from light to dark blue
17. (markdown) "## Residual Analysis"
18. Histogram of (actual - predicted) residuals
    Check: residuals should be approximately normal around 0
19. (markdown) "## Save Best Model"
20. Save XGBoost model with joblib to models/best_model.pkl
21. Print final summary:
    "=== KhazinaSmart Model Summary ===
     Best Model: XGBoost
     RMSE: X | MAE: Y | MAPE: Z% | R2: W
     Model saved to models/best_model.pkl"

All charts: height=500, template='plotly_white'
Every cell must have output visible.
```

---

## PROMPT 3.2 — Hyperparameter Tuning

```
You have the full KhazinaSmart context above.

Create notebooks/03b_hyperparameter_tuning.ipynb.

Use Optuna to tune XGBoost. If Optuna not installed, use GridSearchCV fallback.

With Optuna:
1. Define objective(trial):
   params = {
     'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
     'max_depth': trial.suggest_int('max_depth', 3, 9),
     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
     'subsample': trial.suggest_float('subsample', 0.6, 1.0),
     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
     'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
   }
   Use TimeSeriesSplit(n_splits=3) cross-validation
   Return mean RMSE across folds

2. Run study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=30, show_progress_bar=True)

3. Print best params and best RMSE

4. Retrain XGBoost on full train set with best params

5. Evaluate on test set: RMSE/MAE/MAPE/R2

6. Plot optimization history: optuna.visualization.plot_optimization_history(study)

7. Save tuned model to models/xgb_tuned.pkl

8. Compare: baseline XGBoost vs tuned XGBoost metrics
```

---

## PROMPT 3.3 — Overstock & Stockout Detection Module

```
You have the full KhazinaSmart context above.

Create src/alerts.py — the risk classification engine.

Functions to implement:

1. classify_inventory_risk(current_stock: float, predicted_demand: float) -> str:
   if current_stock > predicted_demand * 1.3: return "Overstock"
   if current_stock < predicted_demand * 0.7: return "Stockout Risk"
   return "Healthy"

2. compute_risk_score(current_stock: float, predicted_demand: float) -> float:
   Returns a score 0-100:
   - 0 = perfectly balanced
   - 100 = critical risk (extreme overstock or stockout)
   Formula: abs(current_stock - predicted_demand) / predicted_demand * 100, capped at 100

3. generate_alerts_dataframe(df_with_predictions: pd.DataFrame) -> pd.DataFrame:
   Input: dataframe with columns [Store, Dept, Date, Weekly_Sales, predicted_demand]
   Adds: status (Overstock/Stockout Risk/Healthy), risk_score, action_needed
   action_needed logic:
     Overstock → "Reduce orders — excess: {int(current - predicted)} units"
     Stockout Risk → "Increase orders — deficit: {int(predicted - current)} units"
     Healthy → "No action needed"
   Returns sorted by risk_score descending

4. get_top_alerts(alerts_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
   Returns top N highest risk items

5. estimate_financial_impact(alerts_df: pd.DataFrame, avg_unit_cost: float = 50.0) -> dict:
   Returns {
     'overstock_cost_mad': sum of excess units * avg_unit_cost,
     'stockout_lost_sales_mad': sum of deficit units * avg_unit_cost,
     'total_risk_mad': above two summed,
     'items_at_risk': count of non-healthy items
   }

Add full docstrings to every function.
```

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 4 — STREAMLIT DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT 4.1 — Main App Entry Point

```
You have the full KhazinaSmart context above.

Create app/streamlit_app.py — the main entry point.

Must include:
1. st.set_page_config(
     page_title="KhazinaSmart",
     page_icon="📦",
     layout="wide",
     initial_sidebar_state="expanded"
   )

2. Custom CSS injected with st.markdown():
   - Background: #0f0f1a (dark)
   - Card background: #1a1a2e
   - Accent color: #e94560
   - Text: white / #cccccc
   - Metric cards with box-shadow and border-radius: 10px
   - Sidebar background: #16213e

3. Sidebar:
   - "KhazinaSmart 📦" as styled header
   - Tagline: "AI-Powered Inventory Intelligence"
   - Horizontal rule
   - Navigation links to all pages
   - Bottom: "IT DAY'Z Hackathon 2026 | ENSA Tanger"

4. Homepage content:
   - Hero section: Large title "KhazinaSmart" + subtitle
   - 3-column feature cards: Predict Demand / Detect Overstock / AI Chatbot
   - "How it works" section: 4 steps with icons
   - Morocco 2030 badge

Make it visually stunning — this is the first thing the jury sees.
```

---

## PROMPT 4.2 — Dashboard Overview Page

```
You have the full KhazinaSmart context above.

Create app/pages/1_Dashboard.py.

This page loads the model predictions and shows:

1. Load models/xgb_tuned.pkl with @st.cache_resource
2. Load data/processed/features_final.csv with @st.cache_data
3. Run predictions on test set
4. Call generate_alerts_dataframe() from src/alerts.py
5. Call estimate_financial_impact()

LAYOUT:
Row 1 — 5 KPI Metric Cards (st.columns(5)):
  - 📦 Total Products Monitored: total unique (Store, Dept) pairs
  - 🔴 Overstock Alerts: count of Overstock items
  - 🟡 Stockout Risks: count of Stockout Risk items
  - 🎯 Model Accuracy: "MAPE: X%" (hardcode from training)
  - 💰 Capital at Risk (MAD): overstock_cost_mad formatted

Row 2 — 2 columns:
  LEFT (60%): Plotly line chart — Actual vs Predicted weekly sales (full test period)
    - Blue line: actual, Orange dashed: predicted
    - Title: "Sales Forecast vs Actual — Test Period"
  RIGHT (40%): Plotly pie chart — Healthy / Overstock / Stockout Risk distribution
    - Colors: #27ae60 / #e94560 / #f39c12
    - Title: "Inventory Health Distribution"

Row 3 — Full width:
  Plotly bar chart — Top 10 stores by total risk score
  Title: "Stores Most at Risk"

Use st.spinner("Loading predictions...") while computing.
All charts: height=400, template='plotly_dark'
```

---

## PROMPT 4.3 — Forecast Page

```
You have the full KhazinaSmart context above.

Create app/pages/2_Forecast.py.

LAYOUT:
Sidebar filters:
  - st.selectbox("Select Store", options=list of all store IDs)
  - st.selectbox("Select Department", options=list of depts for selected store)
  - st.slider("Forecast Weeks Ahead", 4, 12, 8)
  - st.button("Generate Forecast")

Main content when forecast is generated:
1. Filter data for selected Store + Dept
2. Run model prediction for historical + next N weeks
3. CHART 1 (full width): Plotly line chart
   - Blue solid: historical actual sales
   - Orange solid: model fit on training period
   - Orange dashed: future forecast
   - Shaded area: confidence interval (± 1 std of residuals)
   - Vertical dotted line separating history from forecast
   - Title: f"Demand Forecast — Store {store}, Dept {dept}"
4. CHART 2 (full width): Plotly bar chart — weekly predicted demand for next N weeks
5. TABLE: Next N weeks forecast table with columns:
   Week | Date | Predicted Sales | Lower Bound | Upper Bound | Trend
6. st.download_button: Export forecast as CSV

Add st.info box explaining how to interpret the confidence interval.
```

---

## PROMPT 4.4 — Alerts Page

```
You have the full KhazinaSmart context above.

Create app/pages/3_Alerts.py.

LAYOUT:
1. Title: "⚠️ Inventory Risk Alerts"
2. Filters row (st.columns(4)):
   - Status filter: All / Overstock / Stockout Risk / Healthy
   - Store filter: multiselect
   - Min risk score: slider 0-100
   - st.button("Refresh Alerts")

3. Summary metrics row (st.columns(3)):
   - 🔴 Critical (risk > 70): count + total MAD at risk
   - 🟡 Warning (risk 40-70): count
   - 🟢 Healthy: count

4. Main alerts table using st.dataframe() with column_config:
   - Status: colored badge (use st.column_config.SelectboxColumn)
   - Risk Score: st.column_config.ProgressColumn (0-100)
   - Action Needed: text
   Sort by risk_score descending by default

5. Below table: st.download_button to export filtered alerts as CSV

6. Plotly chart: Risk score distribution histogram
   Color by status (red/yellow/green)
   Title: "Risk Score Distribution"

7. Plotly scatter: Store vs Dept colored by risk score (heatmap style)
   Size: risk_score, Color: risk_score (red-yellow-green scale)
```

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 5 — AI CHATBOT (KhazBot)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT 5.1 — Chatbot Backend Module

```
You have the full KhazinaSmart context above.

Create src/chatbot.py — the KhazBot AI engine.

SYSTEM PROMPT for the LLM (store as constant):
"""
You are KhazBot, the AI inventory assistant for KhazinaSmart — an intelligent inventory management platform.
You help business owners understand their stock health and make smart decisions.
You have access to real inventory data provided as context.
ALWAYS reference specific numbers, stores, departments, or products from the data.
Be concise, professional, and actionable.
Format numbers clearly (e.g., "1,250 units", "45,000 MAD").
Answer in the same language the user writes in (French or English).
Never say you don't have data — analyze what's provided and give your best answer.
"""

Functions:

1. format_inventory_context(alerts_df: pd.DataFrame) -> str:
   Converts top 50 rows of alerts_df to a clean string summary:
   - Overall stats: total items, overstock count, stockout count, healthy count
   - Top 5 overstock items (Store, Dept, risk_score, action_needed)
   - Top 5 stockout risk items
   - Total capital at risk

2. answer_inventory_question(question: str, alerts_df: pd.DataFrame, api_key: str = None) -> str:
   - Format context with format_inventory_context()
   - Build messages: [system, user_with_context_and_question]
   - Try Claude API (anthropic library) if api_key provided
   - Fallback: rule-based answers for common questions:
     * "overstock" → list top 5 overstock items
     * "order" or "acheter" → list stockout risks + recommended quantities  
     * "budget" → total capital at risk in MAD
     * "best" or "meilleur" → top performing store by avg sales
     * else → general summary of inventory health
   - Return formatted string response

3. get_starter_questions() -> list:
   Returns: [
     "Which products are most overstocked? 🔴",
     "What should I order this week? 📦",
     "What is the total budget at risk? 💰",
     "Which store has the highest risk? 🏪",
     "Show me a summary of inventory health 📊"
   ]

Add clear docstrings. Handle all exceptions gracefully.
```

---

## PROMPT 5.2 — Chatbot Streamlit Page

```
You have the full KhazinaSmart context above.

Create app/pages/4_Chatbot.py — the KhazBot chat interface.

LAYOUT:
1. Title: "🤖 KhazBot — Your AI Inventory Assistant"
2. Subtitle: "Ask anything about your inventory in French or English"

3. Starter questions row — 5 clickable st.button() pills in st.columns(5):
   Use get_starter_questions() from src/chatbot.py
   When clicked: auto-fill the chat with that question and trigger response

4. Chat interface:
   - Initialize st.session_state.messages = [] if not exists
   - Display all messages with st.chat_message("user") and st.chat_message("assistant")
   - KhazBot avatar: "📦"
   - User avatar: "👤"

5. Chat input:
   prompt = st.chat_input("Ask KhazBot about your inventory...")
   if prompt:
     - Add to session_state.messages
     - Show user message
     - Show st.spinner("KhazBot is analyzing your inventory...")
     - Call answer_inventory_question()
     - Show assistant response with st.chat_message
     - Add response to session_state.messages

6. Sidebar in this page:
   - "Sample Questions" expander with all starter questions listed
   - "Data Summary" expander showing quick stats
   - st.button("Clear Chat History")

Style: dark theme consistent with rest of app.
Make KhazBot responses formatted nicely — use bullet points and bold numbers.
```

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 6 — INTEGRATION & POLISH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT 6.1 — Full Integration & Error Handling

```
You have the full KhazinaSmart context above.

Review and fix all integration issues in the KhazinaSmart Streamlit app.

Specifically:
1. Add @st.cache_data to ALL data loading functions across all pages
2. Add @st.cache_resource to ALL model loading functions
3. Add try/except to ALL prediction calls — show st.error() on failure
4. Add st.spinner() to ALL heavy computation sections
5. Make sure all pages import from src/ correctly using sys.path manipulation:
   import sys, os
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
6. Add a loading screen / progress bar on initial app load
7. Test all 4 pages run without error
8. Ensure dark theme CSS is applied globally (inject in streamlit_app.py)

Output: updated versions of any files that need changes.
```

---

## PROMPT 6.2 — Performance & Final Polish

```
You have the full KhazinaSmart context above.

Polish the KhazinaSmart app for the hackathon demo.

1. Add animated metric cards using custom HTML/CSS in st.markdown()
   Each KPI card should have: icon, value, label, colored border-left

2. Add a "Last Updated" timestamp in the sidebar

3. Add a welcome toast notification: st.toast("Welcome to KhazinaSmart! 📦", icon="🎉")

4. Add page transition header to each page:
   A colored banner with page title and description

5. Make the alerts table rows clickable — when a row is selected, show a details panel:
   - Product details
   - Historical sales sparkline (mini Plotly chart)
   - Recommended action highlighted

6. Add a "Demo Mode" toggle in sidebar that loads sample data if no model is found
   This ensures the demo NEVER crashes even if model file is missing

7. Footer on every page: "KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business"

The goal: the app must look like a real SaaS product, not a student project.
```

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 7 — PITCH & PRESENTATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT 7.1 — 5-Minute Pitch Script

```
You have the full KhazinaSmart context above.

Write a 5-minute hackathon pitch script for KhazinaSmart.

Structure (with timing):
[0:00-0:30] HOOK — Start with a powerful stat or story about a Moroccan shop losing money
[0:30-1:30] PROBLEM — Overstock and stockout crisis for Moroccan SMEs
[1:30-3:00] LIVE DEMO — Script for exactly what to click and say during the demo:
  - "Let me show you the dashboard..."
  - "Notice here we have 127 overstock alerts..."
  - "I'll ask KhazBot: what should I order this week?"
  - "The model predicted with X% accuracy that..."
[3:00-3:45] TECH — 45 seconds explaining ML model, features, accuracy metrics
[3:45-4:15] IMPACT — Morocco 2030, SME competitiveness, cost reduction potential
[4:15-4:45] VISION — "By 2030, every Moroccan startup with KhazinaSmart"
[4:45-5:00] CLOSE — Strong ending + team introduction

Rules:
- Write in English with some French phrases for impact
- Include [DEMO: click X] stage directions
- Bold the most important sentences
- Keep sentences short and punchy
- The jury has seen 20 projects today — be memorable
```

---

## PROMPT 7.2 — Jury Q&A Preparation

```
You have the full KhazinaSmart context above.

Generate 15 tough jury questions for KhazinaSmart with expert answers.

Categories:
- Technical (model choice, accuracy, data)
- Business (market, monetization, scalability)
- Morocco 2030 (alignment, impact, deployment)
- Team (why you, experience, skills)
- Demo (what if it crashes, limitations)

For each question: give a 3-4 sentence answer that is confident, honest, and shows depth.
Format as Q: ... / A: ...
```

---

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMERGENCY PROMPTS (if something breaks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PROMPT E.1 — App Won't Start

```
My KhazinaSmart Streamlit app crashes on startup with this error: [paste error]
Context: [paste KHAZINASMART_CONTEXT.md]
Fix the error. Show me the exact lines to change. Do not rewrite the entire file.
```

---

## PROMPT E.2 — Model Performance Too Low

```
My XGBoost model on the Walmart dataset has MAPE = X% and R2 = Y. 
This is not good enough. 
Context: [paste KHAZINASMART_CONTEXT.md]
Suggest 5 specific improvements I can implement in the next 30 minutes:
- Feature additions
- Hyperparameter adjustments
- Data preprocessing fixes
Show exact code for each improvement.
```

---

## PROMPT E.3 — Generate Sample Data (if Kaggle is unavailable)

```
You have the full KhazinaSmart context above.

Generate synthetic Walmart-like data for demo purposes.
Create generate_sample_data.py that produces:
- 10 stores, 20 departments, 100 weeks of data
- Realistic sales patterns with seasonality (higher in Nov-Dec)
- 3 store types (A: high, B: medium, C: low sales)
- Holiday effect: +20% sales on holiday weeks
- Random markdowns
- Save to data/raw/train.csv, stores.csv, features.csv
This data must work with our existing pipeline without any changes.
```

---

## PROMPT E.4 — Quick Demo Mode (last resort)

```
You have the full KhazinaSmart context above.

Create app/demo_mode.py that hardcodes a complete realistic demo dataset:
- Pre-computed predictions for 45 stores × 20 departments
- Pre-computed alerts (30% overstock, 20% stockout, 50% healthy)
- Pre-computed financial impact: 125,000 MAD at risk
- Chatbot responses hardcoded for the 5 starter questions

This is used when st.session_state.demo_mode = True in the sidebar toggle.
The demo must look 100% real — no "demo" labels anywhere in the UI.
```

---

---

# 📋 QUICK REFERENCE — CLAUDE SESSION STARTERS

## Every new session, start with:
```
I am working on KhazinaSmart, an AI inventory prediction platform for a 48-hour hackathon at ENSA Tanger (IT DAY'Z 2026). Here is the full project context:

[PASTE ENTIRE KHAZINASMART_CONTEXT.md HERE]

Now I need your help with: [describe what you need]
```

## For debugging:
```
[PASTE CONTEXT]
Here is my current code: [paste code]
Here is the error: [paste error]
Fix it. Show exact changes only, no rewrites.
```

## For code review before jury:
```
[PASTE CONTEXT]
Review this code for: 1) bugs 2) performance 3) readability 4) what a jury would notice.
Here is the code: [paste]
```
