# 📱 UnplugAI

**Smart detection of smartphone addiction risk using machine learning.**

Graduation project by **Mohd Abdullah Arif**, AKTU (BRSVPRECB), 2026.

🔗 **Live demo:** http://localhost:8501
                   http://192.168.29.189:8501

---

## Overview

UnplugAI predicts whether a person's smartphone usage pattern indicates addiction. A user enters their daily habits (screen time, social media, gaming, sleep, notifications, app opens, and so on) and the app returns:

- an **addiction risk score** (0–100%) with a Low / Moderate / High label
- an **explanation chart** showing which habits pushed the score up or down

## Results

| Metric | Value |
|---|---|
| Model | XGBoost |
| Validation | 5-fold stratified cross-validation |
| **ROC-AUC (out-of-fold)** | **0.9643** |
| Training rows | 691,369 |

Fold AUCs ranged from 0.9636 to 0.9651, so the model is stable across splits. The strongest predictors were daily screen time, weekend screen time, and social media hours.

## Method

1. **Data:** tabular smartphone-usage data with 9 numeric features and 3 categorical features, plus the target `addicted_label`.
2. **Feature engineering:** 10 extra features, such as social-media share of screen time, screen time per hour of sleep, notifications per app open, and weekend vs weekday difference.
3. **Model:** XGBoost with native categorical support and early stopping.
4. **Evaluation:** out-of-fold AUC across 5 stratified folds.
5. **Explainability:** per-prediction feature contributions (SHAP-style values from XGBoost).
6. **Deployment:** Streamlit web app.

## Project structure

```
├── app.py             # Streamlit web app
├── train.py           # Training + evaluation script
├── model.joblib       # Trained model (created by train.py)
├── meta.json          # Metrics, categories, chart data (created by train.py)
├── requirements.txt   # Python dependencies
└── screenshots/       # App screenshots
```

## Run it locally

```bash
pip install -r requirements.txt
python train.py train.csv      # optional: retrain (needs train.csv)
streamlit run app.py
```

## Dataset

Kaggle Playground Series S6E8 (smartphone usage and addiction prediction), based on the
[Smartphone Usage and Addiction Prediction](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction) dataset.

## Limitations

- The data is synthetic benchmark data, so the results show that the method works, **not** clinical accuracy.
- This tool is an educational demo and **not a medical diagnosis**.

## Future work

- Collect usage data automatically through Android Digital Wellbeing / iOS Screen Time.
- Validate on real survey data.
- Add personalised recommendations for reducing screen time.
