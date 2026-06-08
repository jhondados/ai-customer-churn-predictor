"""Churn predictor with SHAP explanations."""
import xgboost as xgb
import shap
import pandas as pd
import numpy as np
from typing import Dict, List

class ChurnPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc", early_stopping_rounds=50)
        self.explainer = None
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def predict_with_explanation(self, X: pd.DataFrame) -> List[Dict]:
        proba = self.model.predict_proba(X)[:, 1]
        shap_values = self.explainer.shap_values(X)
        results = []
        for i, (prob, shap_row) in enumerate(zip(proba, shap_values)):
            top_factors = sorted(zip(self.feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:5]
            risk_level = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
            intervention = self._recommend_intervention(top_factors, risk_level)
            results.append({"churn_probability": float(prob), "risk_level": risk_level,
                           "top_churn_drivers": [{"feature": f, "impact": float(s)} for f, s in top_factors],
                           "recommended_action": intervention})
        return results

    def _recommend_intervention(self, top_factors: list, risk_level: str) -> str:
        factor_names = [f[0] for f in top_factors]
        if "days_since_last_purchase" in factor_names: return "Send win-back email with 20% discount"
        if "support_tickets_30d" in factor_names: return "Proactive customer success call"
        if "feature_usage_decline" in factor_names: return "Send product tips + onboarding session"
        return "General retention campaign"
