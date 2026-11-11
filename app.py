"""Step 2: the demo app.   Run with:  streamlit run app.py"""
import json
import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb
from train import RAW_NUM, CAT, FEATURES, make_X

st.set_page_config(page_title="UnplugAI", page_icon="📱", layout="wide")


@st.cache_resource
def load():
    model = joblib.load("model.joblib")
    meta = json.load(open("meta.json"))
    return model, meta


model, meta = load()
LABELS = {
    "age": "Age", "daily_screen_time_hours": "Daily screen time (hours)",
    "social_media_hours": "Social media (hours/day)", "gaming_hours": "Gaming (hours/day)",
    "work_study_hours": "Work / study (hours/day)", "sleep_hours": "Sleep (hours/night)",
    "notifications_per_day": "Notifications per day", "app_opens_per_day": "App opens per day",
    "weekend_screen_time": "Weekend screen time (hours/day)",
}

st.title(" UnplugAI")
st.caption("Smart detection of smartphone addiction, powered by Machine Learning (XGBoost)")
tab_pred, tab_perf, tab_about = st.tabs(["🔮 Try it", "📊 Model performance", "ℹ️ About"])
st.caption("Project by Mohd Abdullah Arif, AKTU(BRSVPRECB), 2026")

with tab_pred:
    left, right = st.columns([1, 1])
    values = {}
    with left:
        st.subheader("Your phone habits")
        for c in RAW_NUM:
            lo, med, hi = meta["ranges"][c]
            if c == "age" or c in ("notifications_per_day", "app_opens_per_day"):
                values[c] = st.slider(LABELS[c], int(lo), int(hi), int(med))
            else:
                values[c] = st.slider(LABELS[c], float(lo), float(hi), float(med), step=0.1)
        values["gender"] = st.selectbox("Gender", meta["categories"]["gender"])
        values["stress_level"] = st.selectbox("Stress level", meta["categories"]["stress_level"])
        values["academic_work_impact"] = st.selectbox(
            "Impact on study/work", meta["categories"]["academic_work_impact"])

    row = pd.DataFrame([values])
    X = make_X(row, meta["categories"])
    prob = float(model.predict_proba(X)[0, 1])

    with right:
        st.subheader("Result")
        level = "Low" if prob < 0.33 else "Moderate" if prob < 0.66 else "High"
        icon = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}[level]
        st.metric("Addiction risk", f"{prob * 100:.0f}%", f"{icon} {level} risk", delta_color="off")
        st.progress(min(max(prob, 0.0), 1.0))

        st.markdown("**Why this score?** (bars to the right push risk up, bars to the left push it down)")
        contrib = model.get_booster().predict(
            xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[0][:-1]
        contrib = pd.Series(contrib, index=FEATURES).sort_values(key=abs, ascending=False).head(8)
        st.bar_chart(contrib.sort_values(), horizontal=True)
        st.caption("Educational demo only, not a medical diagnosis.")

with tab_perf:
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-validated AUC", f"{meta['cv_auc']:.3f}")
    c2.metric("Training rows", f"{meta['n_rows']:,}")
    c3.metric("Share labelled addicted", f"{meta['positive_rate'] * 100:.0f}%")
    a, b = st.columns(2)
    with a:
        st.subheader("ROC curve")
        st.line_chart(pd.DataFrame({"False positive rate": meta["roc"]["fpr"],
                                    "True positive rate": meta["roc"]["tpr"]}),
                      x="False positive rate", y="True positive rate")
    with b:
        st.subheader("Most important features")
        imp = pd.Series(meta["importance"]).sort_values(ascending=False).head(10)
        st.bar_chart(imp.sort_values(), horizontal=True)

with tab_about:
    st.markdown("""
**Problem:** predict whether a person's smartphone usage pattern indicates addiction.

**Pipeline:** data cleaning, 10 engineered features (ratios such as social-media share of screen
time, screen time per hour of sleep), XGBoost, evaluated with 5-fold stratified cross-validation
(AUC), with feature-level explanations for every prediction.

**Limitations:** the data is synthetic/benchmark data, so results show the method works, not
clinical accuracy.
""")

