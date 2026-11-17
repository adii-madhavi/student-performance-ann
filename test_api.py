"""Smoke test for the saved model + API.  Run:  python test_api.py"""
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)
good = {"age": 20, "attendance_pct": 82, "study_hours_per_week": 6.5, "previous_score": 71,
        "assignment_completion_pct": 80, "internal_assessment_score": 68, "participation_score": 7,
        "sleep_hours_per_day": 7, "extracurricular_hours_per_week": 4, "failed_subjects": 0}

r = client.post("/predict", json=good)
assert r.status_code == 200, r.text
body = r.json()
assert body["predicted_performance"] in ("Low", "Medium", "High")
assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
assert body["confidence"] == max(body["probabilities"].values())

strong = {**good, "study_hours_per_week": 30, "previous_score": 95, "internal_assessment_score": 92,
          "attendance_pct": 98, "participation_score": 9.5}
weak = {**good, "study_hours_per_week": 2, "previous_score": 40, "internal_assessment_score": 35,
        "attendance_pct": 50, "participation_score": 2, "failed_subjects": 3}
assert client.post("/predict", json=strong).json()["predicted_performance"] == "High"
assert client.post("/predict", json=weak).json()["predicted_performance"] == "Low"

assert client.post("/predict", json={**good, "attendance_pct": 150}).status_code == 422  # out of range
assert client.post("/predict", json={"age": 20}).status_code == 422                     # missing fields
print("example:", body)
print("all API checks passed")
