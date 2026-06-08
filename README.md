# 📉 AI Customer Churn Predictor

[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.934-green)](.) [![Retention](https://img.shields.io/badge/Retention%20Improvement-%2B28%25-blue)](.) [![Revenue](https://img.shields.io/badge/Revenue%20Saved-R%2412M%2Fyear-orange)](.)

> **Explainable churn prediction** with causal ML intervention recommendations. **0.934 AUC-ROC**, identifies top churn drivers with SHAP and automatically triggers personalized retention campaigns. **R$12M/year** saved.

## 🏆 Business Results
- **+28% retention rate** for high-risk customers (vs control group)
- **R$12M/year** in prevented churn revenue
- **0.934 AUC-ROC** — top 10% of customers = 67% of actual churners
- **SHAP explanations** for every prediction — compliance-ready

## 🔄 System Flow
```
Customer Events → Feature Engineering (340 features) → XGBoost Score
              → SHAP Explanation → Causal Intervention Recommendation
              → Campaign Trigger (email/SMS/offer) → Retention Tracking
```
