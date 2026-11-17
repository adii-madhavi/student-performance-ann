# Student Performance Risk Predictor (ANN)

A feed-forward neural network (Keras/TensorFlow) that classifies a student's expected performance as **Low / Medium / High** from 10 academic and behavioural features. It includes a Logistic Regression baseline, a Streamlit app and a FastAPI endpoint.

Full results and concept explanations: **[reports/Evaluation_Report.md](reports/Evaluation_Report.md)** ([PDF](reports/Evaluation_Report.pdf))

## Project structure
```
student-performance-ann/
├── data/student_data.csv              # 15,000 students, 12 columns
├── notebooks/ann_training.ipynb       # EDA → preprocessing → baseline → ANN → evaluation → save
├── models/
│   ├── student_performance.keras      # trained ANN
│   └── scaler.joblib                  # StandardScaler fit on training data (needed at inference)
├── reports/                           # Evaluation_Report.md, metrics.json, plots
├── predictor.py                       # shared inference (load once, predict)
├── app.py                             # Streamlit UI
├── api.py                             # FastAPI JSON endpoint
├── test_api.py                        # smoke test for model + API
└── requirements.txt
```

## Dataset
* **Target:** `performance_category` (Low 4,950 · Medium 5,100 · High 4,950, balanced)
* **Features (10):** `age`, `attendance_pct`, `study_hours_per_week`, `previous_score`, `assignment_completion_pct`, `internal_assessment_score`, `participation_score` (0–10), `sleep_hours_per_day`, `extracurricular_hours_per_week`, `failed_subjects`
* `student_id` is dropped. There are no missing values and no duplicates.

## Preprocessing
Labels are encoded Low = 0, Medium = 1, High = 2. The data is split 70/15/15 (train/val/test), stratified. A `StandardScaler` is fit on the training set only and saved for inference.

## ANN architecture
`Input(10) → Dense(32, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dropout(0.2) → Dense(3, Softmax)`, with 931 parameters.
It was chosen from 4 candidates compared on the validation set. All scored within 0.2 pp of each other, so the smallest two-hidden-layer network was kept.

## Training
Loss: sparse categorical cross-entropy · Adam (lr 0.001) · batch 32 · up to 150 epochs with EarlyStopping on `val_loss` (patience 10, best weights restored). Training stopped after 31 epochs.

## Results (test set, 2,250 students)
| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (baseline) | 0.707 | 0.708 |
| **ANN** | **0.701** | **0.704** |

The ANN matches the linear baseline. The classes are close to linearly separable and overlap heavily, so about 70 % is the data's ceiling. 98 % of errors are between adjacent categories. See the report for the confusion matrix and observations.

## Setup
Requires **Python 3.10–3.12** (TensorFlow has no stable wheel for 3.13+ on Windows yet).
On Windows, keep the virtual-env path short (e.g. `C:\venvs\ann`). Very long paths break TensorFlow's DLL loading.
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Retrain (re-runs the whole notebook and regenerates the model, metrics and plots):
```bash
cd notebooks && jupyter nbconvert --to notebook --execute --inplace ann_training.ipynb
```
Run the web app (http://localhost:8501):
```bash
streamlit run app.py
```
Run the API (interactive docs at http://127.0.0.1:8000/docs):
```bash
uvicorn api:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"age\":20,\"attendance_pct\":82,\"study_hours_per_week\":6.5,\"previous_score\":71,\"assignment_completion_pct\":80,\"internal_assessment_score\":68,\"participation_score\":7,\"sleep_hours_per_day\":7,\"extracurricular_hours_per_week\":4,\"failed_subjects\":0}"
```
→ `{"predicted_performance":"Medium","confidence":0.5327,"probabilities":{"Low":0.3564,"Medium":0.5327,"High":0.1108}}`

Test the model and API:
```bash
python test_api.py
```

## Demo walkthrough (about 5 min)
1. **Data:** open the notebook, show Step 1 (shape, no nulls or duplicates, balanced classes, boxplots: study hours separates the classes best).
2. **Preprocessing:** the 70/15/15 split, and the scaler fit on train only (why scaling matters).
3. **Baseline vs ANN:** the architecture table, then `model.summary()` (931 params).
4. **Training:** the loss and accuracy curves; early stopping at epoch 31; no over-fitting.
5. **Forward and backprop:** the hand-computed forward pass matching `model.predict`; one gradient step lowering the loss.
6. **Evaluation:** the comparison table and confusion matrices; Medium is the hardest class; errors are near misses.
7. **App:** `streamlit run app.py`, enter a strong and a weak student, show the probabilities.
8. **API:** open `/docs`, use "Try it out" on `/predict`, and show the JSON response.
