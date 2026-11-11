"""Step 1: train the model and save everything the app needs.

Usage:  python train.py path/to/train.csv
Creates: model.joblib, meta.json
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 42
TARGET = "addicted_label"
RAW_NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
           "work_study_hours", "sleep_hours", "notifications_per_day",
           "app_opens_per_day", "weekend_screen_time"]
CAT = ["gender", "stress_level", "academic_work_impact"]
ENG = ["social_frac", "gaming_frac", "weekend_ratio", "weekend_delta",
       "screen_per_sleep", "screen_per_work", "notif_per_open",
       "opens_per_hour", "accounted_hours", "n_missing"]
FEATURES = RAW_NUM + ENG + CAT
EPS = 1e-6


def engineer(df):
    out = df.copy()
    s = out["daily_screen_time_hours"]
    out["social_frac"] = out["social_media_hours"] / (s + EPS)
    out["gaming_frac"] = out["gaming_hours"] / (s + EPS)
    out["weekend_ratio"] = out["weekend_screen_time"] / (s + EPS)
    out["weekend_delta"] = out["weekend_screen_time"] - s
    out["screen_per_sleep"] = s / (out["sleep_hours"] + EPS)
    out["screen_per_work"] = s / (out["work_study_hours"] + EPS)
    out["notif_per_open"] = out["notifications_per_day"] / (out["app_opens_per_day"] + EPS)
    out["opens_per_hour"] = out["app_opens_per_day"] / (s + EPS)
    out["accounted_hours"] = s + out["work_study_hours"] + out["sleep_hours"]
    out["n_missing"] = out[RAW_NUM + CAT].isna().sum(axis=1).astype(float)
    return out


def make_X(df, categories):
    X = engineer(df)[FEATURES].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def params():
    return dict(learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                random_state=SEED, n_jobs=-1)


def main(path):
    df = pd.read_csv(path)
    y = df[TARGET].values
    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in CAT}
    X = make_X(df, categories)

    oof = np.zeros(len(df))
    best_iters = []
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        m = xgb.XGBClassifier(n_estimators=2000, eval_metric="auc",
                              early_stopping_rounds=100, **params())
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        best_iters.append(m.best_iteration)
        print(f"Fold {fold}: AUC = {roc_auc_score(y[va], oof[va]):.4f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"Overall CV AUC: {cv_auc:.4f}")

    final = xgb.XGBClassifier(n_estimators=int(np.mean(best_iters) * 1.1), **params())
    final.fit(X, y)
    joblib.dump(final, "model.joblib")

    fpr, tpr, _ = roc_curve(y, oof)
    keep = np.linspace(0, len(fpr) - 1, 200).astype(int)
    imp = dict(zip(FEATURES, final.feature_importances_.astype(float)))
    ranges = {c: [float(df[c].min()), float(df[c].median()), float(df[c].max())]
              for c in RAW_NUM}
    meta = {"cv_auc": float(cv_auc), "categories": categories, "ranges": ranges,
            "importance": imp, "roc": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
            "n_rows": int(len(df)), "positive_rate": float(y.mean())}
    with open("meta.json", "w") as f:
        json.dump(meta, f)
    print("Saved model.joblib and meta.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train.csv")