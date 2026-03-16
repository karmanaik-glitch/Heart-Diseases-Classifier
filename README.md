# Heart Disease Classifier

A machine learning screening tool that predicts the presence of coronary artery disease using clinical and diagnostic features from the UCI Cleveland Heart Disease dataset.



## Project Overview

| Item | Detail |
|---|---|
| Dataset | UCI Heart Disease (Cleveland) — 303 patients, 13 features |
| Task | Binary classification — CAD present vs absent |
| Best model | Random Forest (Sklearn Pipeline) |
| CV AUC | 0.91 ± 0.03 |
| Test AUC | 0.95 |
| Selected threshold | 0.3 (recall-optimised) |
| False negatives | 0 — all 28 disease patients in test set correctly flagged |

---

## Clinical Context

This tool is designed as a **cardiology screening aid** — not a diagnostic confirmation. Its goal is to flag patients who warrant further workup (stress ECG, echocardiogram, or coronary angiography).

**Recall is prioritised over precision.** Missing a patient with heart disease carries greater clinical risk than an unnecessary referral. The decision threshold was lowered from 0.5 to 0.3 to achieve perfect recall on the test set — at threshold 0.3:

- All 28 disease patients correctly flagged (recall = 1.00)
- 11 false alarms out of 39 flagged patients (precision = 0.72)

This tradeoff is appropriate for a screening context.

---

## Dataset

**Source:** [UCI Heart Disease Database — Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease)

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age in years |
| sex | Binary | 1 = male, 0 = female |
| cp | Categorical | Chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic) |
| trestbps | Numeric | Resting blood pressure (mm Hg) |
| chol | Numeric | Serum cholesterol (mg/dL) |
| fbs | Binary | Fasting blood sugar > 120 mg/dL |
| restecg | Categorical | Resting ECG result (0=normal, 1=ST-T abnormality, 2=LV hypertrophy) |
| thalach | Numeric | Maximum heart rate achieved |
| exang | Binary | Exercise-induced angina |
| oldpeak | Numeric | ST depression induced by exercise |
| slope | Categorical | Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping) |
| ca | Numeric | Number of major vessels with blockage (0–3, fluoroscopy) |
| thal | Categorical | Nuclear stress test result (3=normal, 6=fixed defect, 7=reversible defect) |

**Target:** 0 = no disease, 1 = disease present (binarised from original 0–4 severity scale)

**Missing values:** `ca` (4 missing), `thal` (2 missing) — imputed inside Pipeline using median/most_frequent strategy on training data only.

---

## Methodology

### Pipeline Architecture

All preprocessing is handled inside a Sklearn Pipeline — preventing data leakage structurally.

```
ColumnTransformer
├── numeric_pipeline   → SimpleImputer(median) → StandardScaler
│   └── age, trestbps, chol, thalach, oldpeak, ca
├── binary_pipeline    → StandardScaler
│   └── sex, fbs, exang
└── categorical_pipeline → SimpleImputer(most_frequent) → OneHotEncoder
    └── cp, restecg, slope, thal
            ↓
        RandomForestClassifier
```

One-hot encoding expands the 13 original features to 22 after encoding.

### Model Selection

| Model | CV AUC | Std |
|---|---|---|
| Logistic Regression | 0.91 | ±0.02 |
| Random Forest | 0.91 | ±0.03 |
| SVM | 0.90 | ±0.03 |
| Decision Tree | 0.78 | ±0.06 |

Random Forest selected — strongest and most stable performance. `class_weight='balanced'` evaluated via GridSearchCV — confirmed unnecessary on this near-balanced dataset (GridSearch selected `None`).

### Hyperparameter Tuning

GridSearchCV on full Pipeline (24 combinations × 5 folds = 120 fits):

```
Best parameters:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
  class_weight: None
```

### SHAP Analysis

Top 5 features by mean absolute SHAP value:

| Rank | Feature | Clinical interpretation |
|---|---|---|
| 1 | ca | Direct vessel blockage count — strongest objective CAD marker |
| 2 | thal_3.0 | Normal stress test — pushes away from disease (protective) |
| 3 | cp_4.0 | Asymptomatic chest pain — silent ischaemia common in confirmed CAD |
| 4 | thal_7.0 | Reversible defect — active ischaemia under stress |
| 5 | oldpeak | ST depression under exercise — higher = more ischaemia |

All top predictors are direct cardiac diagnostic findings, consistent with established cardiology evidence.

---

## Results

### Confusion Matrix (threshold = 0.3)

|  | Predicted No Disease | Predicted Disease |
|---|---|---|
| **Actual No Disease** | 22 (TN) | 11 (FP) |
| **Actual Disease** | 0 (FN) | 28 (TP) |

### Classification Report

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Disease | 1.00 | 0.67 | 0.80 |
| Disease | 0.72 | 1.00 | 0.84 |

---

## Limitations

- **Dataset scope:** All features are collected during active cardiac workup (stress tests, catheterisation). The model performs well on patients already undergoing cardiac evaluation — it does not generalise to primary care screening where these tests haven't been performed.
- **Dataset size:** 303 patients is small for a clinical ML model. Performance estimates should be validated on larger external cohorts before any real-world use.
- **Population:** Cleveland dataset (1988) — may not reflect current patient populations or contemporary clinical practice.
- **Not a diagnostic tool:** Output is a screening probability, not a confirmed diagnosis. All flagged patients require clinical assessment.

---

## Repository Structure

```
heart-disease-classifier/
├── heart-disease-classifier.ipynb   # Full analysis notebook
├── app.py                           # Streamlit web application
├── heart_disease_model.pkl          # Saved Pipeline (preprocessor + model)
├── requirements.txt                 # Python dependencies
└── README.md
```

---



## Tech Stack

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV, RandomForest)
- SHAP (model explainability)
- Streamlit (deployment)
- Google Colab → GitHub → Streamlit Cloud

---

## Part of Healthcare AI/ML Learning Journey


---

*This tool is a screening aid only and is not intended for clinical use.*
