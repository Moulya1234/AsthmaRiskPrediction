"""Train and compare four classifiers on the original asthma dataset."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "asthma_disease_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "asthma_models.joblib"
METADATA_PATH = MODEL_DIR / "asthma_model_metadata.json"
TARGET = "Diagnosis"
CATEGORICAL = ["Gender", "Ethnicity", "EducationLevel"]
EXCLUDED = ["PatientID", "DoctorInCharge", TARGET]
SOURCE = "https://doi.org/10.34740/KAGGLE/DSV/8669080"


def make_preprocessor(numeric: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, CATEGORICAL)],
        remainder="drop",
    )


def score_model(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    prediction = (probabilities >= 0.5).astype("int8")
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Original asthma dataset missing. Run `python download_dataset.py` first.")
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain a {TARGET!r} column.")
    features = [column for column in data.columns if column not in EXCLUDED]
    missing = [column for column in CATEGORICAL if column not in features]
    if missing:
        raise ValueError(f"Expected categorical columns are missing: {missing}")
    numeric = [column for column in features if column not in CATEGORICAL]
    X = data[features].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(data[TARGET], errors="coerce")
    valid = y.notna() & y.isin([0, 1])
    X, y = X.loc[valid], y.loc[valid].astype("int8")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    }
    fitted_models = {}
    evaluation = {}
    for name, classifier in classifiers.items():
        pipeline = Pipeline(
            [("preprocess", make_preprocessor(numeric)), ("classifier", classifier)]
        )
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        evaluation[name] = score_model(y_test, probability)
        fitted_models[name] = pipeline

    best_model = max(evaluation, key=lambda name: evaluation[name]["roc_auc"])
    print(f"Rows used: {len(y):,} | positive asthma labels: {int(y.sum()):,} ({y.mean():.1%})")
    print("Stratified holdout: 80% train / 20% test; random_state=42; class threshold=0.50")
    for name, metrics in evaluation.items():
        print(
            f"{name:23s} accuracy={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} F1={metrics['f1']:.3f} AUC={metrics['roc_auc']:.3f} "
            f"AP={metrics['average_precision']:.3f} CM={metrics['confusion_matrix']}"
        )
    print(f"Highest holdout ROC-AUC: {best_model}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models, MODEL_PATH)
    metadata = {
        "features": features,
        "numeric_features": numeric,
        "categorical_features": CATEGORICAL,
        "target_column": TARGET,
        "target_definition": "0 = source record labeled no asthma; 1 = source record labeled asthma.",
        "probability_threshold": 0.5,
        "risk_bands": {"low_max_exclusive": 0.35, "moderate_max_exclusive": 0.65},
        "rows_used": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "test_size": 0.2,
        "best_model_by_roc_auc": best_model,
        "evaluation": evaluation,
        "source": SOURCE,
        "limitations": "The Kaggle publisher identifies this dataset as synthetic. It has a small positive class (about 5%). Class weighting permits positive predictions but does not create reliable signal. Holdout performance is weak; score bands are illustrative, uncalibrated model scores, not medical risk probabilities.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model bundle: {MODEL_PATH}")
    print(f"Saved evaluation metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
