"""
Universal grocery forecasting model with transfer learning.
Pre-trained on Walmart retail data, fine-tunable on any grocery dataset.
"""
import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_adapter import FEATURE_COLS, build_model_features, standardize, detect_columns

BASE_MODEL_PATHS = [
    os.path.join("models", "xgb_tuned.pkl"),
    os.path.join("models", "best_model.pkl"),
]


class UniversalForecastModel:
    """
    XGBoost demand forecasting model with transfer learning support.
    Loads a Walmart-pretrained base, then fine-tunes on any new grocery dataset.
    """

    def __init__(self):
        self.base_model = self._load_base()
        self.model = None
        self.feature_cols = FEATURE_COLS
        self.metrics = {}
        self.is_transfer = self.base_model is not None

    def _load_base(self):
        for p in BASE_MODEL_PATHS:
            if os.path.exists(p):
                try:
                    return joblib.load(p)
                except Exception:
                    pass
        return None

    def fit(self, df_standard: pd.DataFrame, n_transfer_trees: int = 80) -> dict:
        """
        Fit (or transfer-learn) on standardized grocery dataframe.
        Returns dict with train/test metrics.
        """
        df_feat = build_model_features(df_standard)
        avail = [c for c in self.feature_cols if c in df_feat.columns]

        cutoff = df_feat["date"].quantile(0.8)
        train = df_feat[df_feat["date"] <= cutoff]
        test  = df_feat[df_feat["date"] >  cutoff]

        X_tr, y_tr = train[avail].fillna(0), train["sales"]
        X_te, y_te = test[avail].fillna(0),  test["sales"]

        if self.is_transfer and self.base_model is not None:
            # XGBoost transfer learning: warm-start from Walmart base booster
            try:
                base_booster = self.base_model.get_booster()
                self.model = XGBRegressor(
                    n_estimators=n_transfer_trees,
                    max_depth=6, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0,
                )
                self.model.fit(X_tr, y_tr, xgb_model=base_booster, verbose=False)
                transfer_used = True
            except Exception:
                transfer_used = False
        else:
            transfer_used = False

        if not transfer_used:
            self.model = XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            self.model.fit(X_tr, y_tr, verbose=False)

        preds = self.model.predict(X_te)
        rmse  = np.sqrt(mean_squared_error(y_te, preds))
        mae   = mean_absolute_error(y_te, preds)
        mape  = float(np.mean(np.abs((y_te.values - preds) / (y_te.values + 1))) * 100)
        r2    = r2_score(y_te, preds)

        self.metrics = {
            "rmse": round(rmse, 1), "mae": round(mae, 1),
            "mape": round(mape, 2), "r2": round(r2, 4),
            "transfer_learning": transfer_used,
            "train_rows": len(X_tr), "test_rows": len(X_te),
        }

        # Attach predictions back to test slice
        test_with_preds = test.copy()
        test_with_preds["predicted"] = preds
        self._test_df = test_with_preds
        self._full_df = df_feat.copy()
        self.feature_importance = dict(zip(avail, self.model.feature_importances_))

        return self.metrics

    def predict_future(self, df_standard: pd.DataFrame, weeks: int = 8) -> pd.DataFrame:
        """
        Predict `weeks` weeks into the future for each (store, category) pair.
        Returns dataframe with columns: store_id, category, date, predicted, lower, upper.
        """
        if self.model is None or not hasattr(self, "_full_df"):
            raise ValueError("Model not fitted yet.")

        avail = [c for c in self.feature_cols if c in self._full_df.columns]
        results = []

        for (store, cat), grp in self._full_df.groupby(["store_id", "category"]):
            grp = grp.sort_values("date")
            last = grp.iloc[-1]
            lag1, lag2, lag4, lag8 = (
                last.get("sales", 0), grp["sales"].iloc[-2] if len(grp) > 1 else last["sales"],
                grp["sales"].iloc[-4] if len(grp) > 3 else last["sales"],
                grp["sales"].iloc[-8] if len(grp) > 7 else last["sales"],
            )
            roll4  = grp["sales"].tail(4).mean()
            roll12 = grp["sales"].tail(12).mean()
            std4   = grp["sales"].tail(4).std() if len(grp) > 1 else roll4 * 0.1

            last_date = pd.to_datetime(last["date"])
            for w in range(1, weeks + 1):
                fdate = last_date + pd.Timedelta(weeks=w)
                row = {
                    "week_of_year": fdate.isocalendar()[1],
                    "month": fdate.month, "quarter": (fdate.month - 1) // 3 + 1,
                    "year": fdate.year, "is_month_start": int(fdate.day <= 7),
                    "is_month_end": int(fdate.day >= 24),
                    "is_promoted": 0,
                    "lag_1": lag1, "lag_2": lag2, "lag_4": lag4, "lag_8": lag8,
                    "roll_mean_4": roll4, "roll_mean_12": roll12, "roll_std_4": std4,
                    "store_code": last.get("store_code", 0),
                    "category_code": last.get("category_code", 0),
                }
                x = pd.DataFrame([{c: row.get(c, 0) for c in avail}])
                pred = float(self.model.predict(x)[0])
                pred = max(0, pred)
                results.append({
                    "store_id": store, "category": cat,
                    "date": fdate, "predicted": round(pred, 2),
                    "lower": round(max(0, pred - std4), 2),
                    "upper": round(pred + std4, 2),
                })
                lag8, lag4, lag2, lag1 = lag4, lag2, lag1, pred
                roll4  = (roll4 * 3 + pred) / 4
                roll12 = (roll12 * 11 + pred) / 12

        return pd.DataFrame(results)

    def get_predictions(self) -> pd.DataFrame:
        """Return test-set predictions from the last fit call."""
        if not hasattr(self, "_test_df"):
            return pd.DataFrame()
        return self._test_df

    def get_full_df(self) -> pd.DataFrame:
        if not hasattr(self, "_full_df"):
            return pd.DataFrame()
        return self._full_df
