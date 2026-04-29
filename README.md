# KhazinaSmart 📦
### AI-Powered Inventory Intelligence Platform

## 🏆 Hackathon Achievement

**IT DAY'Z 11th Edition Hackathon** — ENSA Tanger, April 18-19, 2026  
**Theme:** AI for Startups & Business  
**Result:** 🥈 2nd Place

Built in 48 hours by a team of 3 Master's students — M2 Intelligence Artificielle pour l'Économie Numérique et la Gestion

---

## What is KhazinaSmart?

KhazinaSmart (Arabic: خزينة = treasury/stock + Smart) helps startups, SMEs, and retail businesses:
- **Predict demand** for coming weeks using XGBoost ML
- **Detect overstock** before products expire or freeze capital
- **Detect stockout risk** before sales are lost
- **Generate purchase recommendations** automatically
- **Answer natural language questions** about inventory via KhazBot AI chatbot

---

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app/streamlit_app.py
```

## Run Notebooks (in order)

1. `notebooks/01_data_loading.ipynb` — Load & clean Walmart data
2. `notebooks/02_EDA.ipynb` — Exploratory data analysis
3. `notebooks/02b_feature_engineering.ipynb` — Feature engineering
4. `notebooks/03_model_training.ipynb` — Train XGBoost model
5. `notebooks/03b_hyperparameter_tuning.ipynb` — Tune hyperparameters

## Project Structure

```
KhazinaSmart/
├── data/raw/           # Walmart CSVs (train.csv, stores.csv, features.csv)
├── data/processed/     # Cleaned & engineered data
├── notebooks/          # Jupyter notebooks (run in order)
├── src/                # Python modules
├── app/                # Streamlit app
│   └── pages/          # Multi-page app
├── models/             # Trained model files (.pkl)
└── reports/            # Analysis outputs
```

## Dataset

Walmart Store Sales Forecasting (Kaggle) — 45 stores, ~420K rows, 143 weeks (2010–2012).

---

*KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business*
