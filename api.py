"""Prediction API.  Run:  uvicorn api:app --reload   →  docs at http://127.0.0.1:8000/docs"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predictor import predict


class Student(BaseModel):
    age: int = Field(..., ge=10, le=100, examples=[20])
    attendance_pct: float = Field(..., ge=0, le=100, examples=[82])
    study_hours_per_week: float = Field(..., ge=0, le=100, examples=[6.5])
    previous_score: float = Field(..., ge=0, le=100, examples=[71])
    assignment_completion_pct: float = Field(..., ge=0, le=100, examples=[80])
    internal_assessment_score: float = Field(..., ge=0, le=100, examples=[68])
    participation_score: float = Field(..., ge=0, le=10, examples=[7])
    sleep_hours_per_day: float = Field(..., ge=0, le=24, examples=[7])
    extracurricular_hours_per_week: float = Field(..., ge=0, le=100, examples=[4])
    failed_subjects: int = Field(..., ge=0, le=20, examples=[0])


app = FastAPI(title="Student Performance Risk Predictor (ANN)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(student: Student):
    return predict(student.model_dump())
