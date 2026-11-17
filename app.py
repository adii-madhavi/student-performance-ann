"""Prediction UI.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from predictor import CLASSES, predict

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Risk Predictor")
st.caption("Feed-forward ANN (Keras) — predicts Low / Medium / High performance")

with st.form("student"):
    c1, c2 = st.columns(2)
    s = {
        "study_hours_per_week": c1.number_input("Study hours / week", 0.0, 100.0, 6.5, 0.5),
        "attendance_pct": c2.number_input("Attendance (%)", 0.0, 100.0, 82.0, 1.0),
        "previous_score": c1.number_input("Previous score", 0.0, 100.0, 71.0, 1.0),
        "assignment_completion_pct": c2.number_input("Assignment completion (%)", 0.0, 100.0, 80.0, 1.0),
        "internal_assessment_score": c1.number_input("Internal assessment score", 0.0, 100.0, 68.0, 1.0),
        "participation_score": c2.number_input("Participation (0–10)", 0.0, 10.0, 7.0, 0.1),
        "sleep_hours_per_day": c1.number_input("Sleep hours / day", 0.0, 24.0, 7.0, 0.1),
        "extracurricular_hours_per_week": c2.number_input("Extracurricular hours / week", 0.0, 100.0, 4.0, 0.5),
        "age": c1.number_input("Age", 10, 100, 20, 1),
        "failed_subjects": c2.number_input("Failed subjects", 0, 20, 0, 1),
    }
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    r = predict(s)
    colour = {"Low": "red", "Medium": "orange", "High": "green"}[r["predicted_performance"]]
    st.markdown(f"### Predicted performance: :{colour}[**{r['predicted_performance']}**]")
    st.metric("Prediction probability", f"{r['confidence']:.1%}")
    st.bar_chart(pd.Series(r["probabilities"], name="probability").reindex(CLASSES))
    st.json(r)
