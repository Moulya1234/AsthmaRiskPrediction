"""Multi-page respiratory health risk recognition demonstration."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "asthma_disease_data.csv"
MODEL_PATH = ROOT / "models" / "asthma_models.joblib"
METADATA_PATH = ROOT / "models" / "asthma_model_metadata.json"

st.set_page_config(
    page_title="Respiratory Health Risk Recognition",
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_NAMES = [
    "Dashboard", "Asthma Prediction", "Dataset Analysis", "Algorithms",
    "Model Performance", "Confusion Matrix", "Feature Analysis", "About",
]
FEATURE_LABELS = {
    "Age": "Age",
    "Gender": "Gender code",
    "Ethnicity": "Ethnicity code",
    "EducationLevel": "Education level code",
    "BMI": "BMI",
    "Smoking": "Smoking",
    "PhysicalActivity": "Physical activity (dataset scale)",
    "DietQuality": "Diet quality (dataset scale)",
    "SleepQuality": "Sleep quality (dataset scale)",
    "PollutionExposure": "Pollution exposure (dataset scale)",
    "PollenExposure": "Pollen exposure (dataset scale)",
    "DustExposure": "Dust exposure (dataset scale)",
    "PetAllergy": "Pet allergy",
    "FamilyHistoryAsthma": "Family history of asthma",
    "HistoryOfAllergies": "History of allergies",
    "Eczema": "Eczema",
    "HayFever": "Hay fever",
    "GastroesophagealReflux": "Gastroesophageal reflux",
    "LungFunctionFEV1": "Lung function FEV1 (source units)",
    "LungFunctionFVC": "Lung function FVC (source units)",
    "Wheezing": "Wheezing",
    "ShortnessOfBreath": "Shortness of breath",
    "ChestTightness": "Chest tightness",
    "Coughing": "Coughing",
    "NighttimeSymptoms": "Nighttime symptoms",
    "ExerciseInduced": "Exercise-induced symptoms",
}
BINARY_FIELDS = [
    "Smoking", "PetAllergy", "FamilyHistoryAsthma", "HistoryOfAllergies", "Eczema",
    "HayFever", "GastroesophagealReflux", "Wheezing", "ShortnessOfBreath",
    "ChestTightness", "Coughing", "NighttimeSymptoms", "ExerciseInduced",
]
CATEGORICAL_FIELDS = ["Gender", "Ethnicity", "EducationLevel"]
UNITS = {
    "Age": "years", "BMI": "kg/m²", "Gender": "source code", "Ethnicity": "source code",
    "EducationLevel": "source code", "Smoking": "No / Yes", "PetAllergy": "No / Yes",
    "FamilyHistoryAsthma": "No / Yes", "HistoryOfAllergies": "No / Yes", "Eczema": "No / Yes",
    "HayFever": "No / Yes", "GastroesophagealReflux": "No / Yes", "Wheezing": "No / Yes",
    "ShortnessOfBreath": "No / Yes", "ChestTightness": "No / Yes", "Coughing": "No / Yes",
    "NighttimeSymptoms": "No / Yes", "ExerciseInduced": "No / Yes",
    "PhysicalActivity": "dataset scale", "DietQuality": "dataset scale", "SleepQuality": "dataset scale",
    "PollutionExposure": "dataset scale", "PollenExposure": "dataset scale", "DustExposure": "dataset scale",
    "LungFunctionFEV1": "not specified by source", "LungFunctionFVC": "not specified by source",
}


@st.cache_data
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_models(path: str, modified_time: float):
    return joblib.load(path)


@st.cache_data
def load_metadata(path: str, modified_time: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def field_label(column: str) -> str:
    return FEATURE_LABELS.get(column, column)


def metric_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": metrics["accuracy"]},
            {"Metric": "Precision", "Value": metrics["precision"]},
            {"Metric": "Recall / sensitivity", "Value": metrics["recall"]},
            {"Metric": "F1-score", "Value": metrics["f1"]},
            {"Metric": "Specificity", "Value": metrics["specificity"]},
            {"Metric": "ROC-AUC", "Value": metrics["roc_auc"]},
            {"Metric": "Average precision", "Value": metrics["average_precision"]},
        ]
    )


if not DATA_PATH.exists() or not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Project files are missing. Run `python download_dataset.py` and then `python train_model.py`.")
    st.stop()

data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
models = load_models(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
metadata = load_metadata(str(METADATA_PATH), METADATA_PATH.stat().st_mtime)
features = metadata["features"]
evaluation = metadata["evaluation"]
with st.sidebar:
    st.title("Respiratory health", icon=":material/air:")
    st.caption("Asthma risk recognition · academic demo")
    st.radio("Navigate", PAGE_NAMES, index=1, key="project_navigation", label_visibility="visible")
    selected_page = st.session_state.project_navigation
    st.caption("Synthetic dataset · not a clinical tool")


if selected_page == "Dashboard":
    st.title("Project dashboard", icon=":material/dashboard:")
    st.write("Explore the asthma prediction demo, the source dataset, and four trained classifiers.")
    best = metadata["best_model_by_roc_auc"]
    with st.container(horizontal=True):
        st.metric("Dataset records", f"{metadata['rows_used']:,}", border=True)
        st.metric("Asthma-labeled records", f"{metadata['positive_count']:,} ({metadata['positive_prevalence']:.1%})", border=True)
        st.metric("Input features", str(len(features)), border=True)
        st.metric("Top holdout ROC-AUC", f"{evaluation[best]['roc_auc']:.3f}", help=best, border=True)

    overview, labels = st.columns([1.2, 1])
    with overview:
        with st.container(border=True):
            st.subheader("How the app works", divider="gray")
            st.markdown("**Profile inputs → preprocessing → four classifiers → dataset label and score.**")
            st.info("Choose **Asthma Prediction** in the sidebar to enter a profile. The form starts with an example source record labeled asthma.", icon=":material/lightbulb:")
    with labels:
        with st.container(border=True):
            st.subheader("Target labels", divider="gray")
            st.success("**1 — Asthma** · source dataset label")
            st.info("**0 — No asthma** · source dataset label")
    st.warning("The source dataset is synthetic and holdout performance is weak. Scores and risk bands are educational outputs, not medical probabilities.", icon=":material/warning:")


elif selected_page == "Asthma Prediction":
    st.title("Asthma prediction", icon=":material/health_and_safety:")
    st.write("Enter the profile and exposure attributes used by the asthma dataset.")
    st.warning("This synthetic-data model is not validated for screening or diagnosis. Seek professional care for health concerns.", icon=":material/medical_information:")

    train_indices, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        data.index, test_size=metadata["test_size"], random_state=42, stratify=data[metadata["target_column"]]
    )
    train_positive = data.loc[train_indices]
    train_positive = train_positive[train_positive[metadata["target_column"]] == 1]
    reference_model = models[best := metadata["best_model_by_roc_auc"]]
    ref_scores = reference_model.predict_proba(train_positive[features])[:, 1]
    default_row = train_positive.iloc[int(np.argmax(ref_scores))]

    with st.form("asthma_prediction_form", border=True):
        st.caption("Starting values come from a source training record labeled asthma. They are only an example profile.")
        with st.container(border=True):
            st.subheader("Personal details")
            cols = st.columns(3)
            age_values = pd.to_numeric(data["Age"], errors="coerce").dropna()
            bmi_values = pd.to_numeric(data["BMI"], errors="coerce").dropna()
            age = cols[0].number_input("Age (years)", min_value=int(age_values.min()), max_value=int(age_values.max()), value=int(default_row["Age"]))
            bmi = cols[1].number_input("BMI (kg/m²)", min_value=float(bmi_values.min()), max_value=float(bmi_values.max()), value=float(default_row["BMI"]), step=0.1)
            values = {"Age": int(age), "BMI": float(bmi)}
            for col, field in zip(cols[2:], CATEGORICAL_FIELDS[:1]):
                choices = sorted(pd.to_numeric(data[field], errors="coerce").dropna().astype(int).unique().tolist())
                current = int(default_row[field])
                values[field] = col.selectbox(field_label(field), choices, index=choices.index(current) if current in choices else 0, key=f"form_{field}")
            cols = st.columns(2)
            for col, field in zip(cols, CATEGORICAL_FIELDS[1:]):
                choices = sorted(pd.to_numeric(data[field], errors="coerce").dropna().astype(int).unique().tolist())
                current = int(default_row[field])
                values[field] = col.selectbox(field_label(field), choices, index=choices.index(current) if current in choices else 0, key=f"form_{field}")

        with st.container(border=True):
            st.subheader("Lifestyle and environment")
            numeric_lifestyle = ["PhysicalActivity", "DietQuality", "SleepQuality", "PollutionExposure", "PollenExposure", "DustExposure"]
            for start in range(0, len(numeric_lifestyle), 3):
                cols = st.columns(3)
                for col, field in zip(cols, numeric_lifestyle[start : start + 3]):
                    observed = pd.to_numeric(data[field], errors="coerce").dropna()
                    values[field] = col.number_input(
                        field_label(field), min_value=float(observed.min()), max_value=float(observed.max()),
                        value=float(default_row[field]), step=0.1, key=f"form_{field}",
                    )
            cols = st.columns(4)
            for col, field in zip(cols, ["Smoking"]):
                values[field] = col.selectbox(field_label(field), [0, 1], index=int(default_row[field]), format_func=lambda x: "Yes" if x else "No", key=f"form_{field}")

        with st.container(border=True):
            st.subheader("Health history")
            history = ["PetAllergy", "FamilyHistoryAsthma", "HistoryOfAllergies", "Eczema", "HayFever", "GastroesophagealReflux"]
            cols = st.columns(3)
            for index, field in enumerate(history):
                col = cols[index % 3]
                values[field] = col.selectbox(field_label(field), [0, 1], index=int(default_row[field]), format_func=lambda x: "Yes" if x else "No", key=f"form_{field}")

        with st.container(border=True):
            st.subheader("Lung function and respiratory symptoms")
            cols = st.columns(2)
            for col, field in zip(cols, ["LungFunctionFEV1", "LungFunctionFVC"]):
                observed = pd.to_numeric(data[field], errors="coerce").dropna()
                values[field] = col.number_input(
                    field_label(field), min_value=float(observed.min()), max_value=float(observed.max()),
                    value=float(default_row[field]), step=0.1, key=f"form_{field}",
                )
            symptoms = ["Wheezing", "ShortnessOfBreath", "ChestTightness", "Coughing", "NighttimeSymptoms", "ExerciseInduced"]
            cols = st.columns(3)
            for index, field in enumerate(symptoms):
                col = cols[index % 3]
                values[field] = col.selectbox(field_label(field), [0, 1], index=int(default_row[field]), format_func=lambda x: "Yes" if x else "No", key=f"form_{field}")

        algorithm = st.selectbox("Choose classifier", list(models), index=list(models).index(best), help=f"{best} had the strongest ROC-AUC on this fixed holdout split.")
        submitted = st.form_submit_button("Estimate risk level", type="primary", width="stretch")

    if submitted:
        input_frame = pd.DataFrame([values], columns=features)
        comparisons = []
        chosen_score = 0.0
        chosen_label = 0
        for name, fitted in models.items():
            score = float(fitted.predict_proba(input_frame)[0, 1])
            label = int(score >= metadata["probability_threshold"])
            comparisons.append({"Classifier": name, "Prediction": "Asthma" if label else "No asthma", "Label": label, "Model score for 1": score})
            if name == algorithm:
                chosen_score, chosen_label = score, label

        bands = metadata["risk_bands"]
        if chosen_score < bands["low_max_exclusive"]:
            level = "Low"
            level_message = st.success
        elif chosen_score < bands["moderate_max_exclusive"]:
            level = "Moderate"
            level_message = st.warning
        else:
            level = "High"
            level_message = st.error

        with st.container(border=True):
            st.subheader("Prediction result")
            level_message(f"Risk level: {level}", icon=":material/monitor_heart:")
            result, score = st.columns(2)
            result.metric("Predicted dataset label", "1 — Asthma" if chosen_label else "0 — No asthma", border=True)
            score.metric("Model score for label 1", f"{chosen_score:.1%}", border=True)
            st.progress(chosen_score)
            st.caption(f"Selected model: {algorithm}. Risk bands: Low < {bands['low_max_exclusive']:.0%}; Moderate < {bands['moderate_max_exclusive']:.0%}; High ≥ {bands['moderate_max_exclusive']:.0%}. This score is not a calibrated disease probability.")
        st.subheader("Classifier cross-comparison", divider="gray")
        comparison_df = pd.DataFrame(comparisons)
        comparison_df["Model score for 1"] = comparison_df["Model score for 1"].map(lambda x: f"{x:.1%}")
        st.dataframe(comparison_df, hide_index=True, width="stretch")
        st.info("Dataset target: 0 = source row labeled no asthma; 1 = source row labeled asthma.")


elif selected_page == "Dataset Analysis":
    st.title("Dataset analysis", icon=":material/database:")
    st.write("Summary statistics for the original asthma dataset used in training.")
    positives = int((data[metadata["target_column"]] == 1).sum())
    a, b, c = st.columns(3)
    a.metric("Records", f"{len(data):,}", border=True)
    b.metric("No asthma label (0)", f"{len(data) - positives:,}", border=True)
    c.metric("Asthma label (1)", f"{positives:,} ({positives / len(data):.1%})", border=True)
    summary_rows = []
    for field in features:
        values = pd.to_numeric(data[field], errors="coerce")
        summary_rows.append({
            "Feature": field_label(field),
            "Unit / coding": UNITS.get(field, "source scale"),
            "Mean": values.mean(),
            "Median": values.median(),
            "Min": values.min(),
            "Max": values.max(),
            "Missing": int(data[field].isna().sum()),
            "Zero values": int((values == 0).sum()),
        })
    st.dataframe(pd.DataFrame(summary_rows).round(3), hide_index=True, width="stretch")
    st.caption("Zero values are not automatically treated as missing: for binary fields, 0 means No. Only blank/non-finite values are median/mode imputed.")
    st.subheader("Diagnosis balance")
    st.bar_chart(data[metadata["target_column"]].value_counts().sort_index().rename(index={0: "No asthma (0)", 1: "Asthma (1)"}))


elif selected_page == "Algorithms":
    st.title("Pattern-recognition algorithms", icon=":material/account_tree:")
    descriptions = {
        "Logistic Regression": "A scaled linear classifier with balanced class weights. It returns a model score for the asthma class.",
        "K-Nearest Neighbors": "Classifies a record from the labels of nearby records in the scaled feature space. The rare positive class can be difficult for KNN to detect.",
        "Decision Tree": "Learns a set of branching rules and is limited to depth five to reduce complexity. Class weights account for the imbalanced labels.",
        "Support Vector Machine": "Uses an RBF kernel to form a non-linear boundary. Class weighting is used because positive labels are rare.",
    }
    for name, detail in descriptions.items():
        metrics = evaluation[name]
        with st.container(border=True):
            st.subheader(name)
            st.write(detail)
            st.caption(f"Holdout ROC-AUC: {metrics['roc_auc']:.3f} · Recall: {metrics['recall']:.1%} · Precision: {metrics['precision']:.1%}")


elif selected_page == "Model Performance":
    st.title("Model performance comparison", icon=":material/monitoring:")
    st.write("All results use the same stratified 80/20 holdout split (random seed 42), at a class threshold of 0.50.")
    rows = []
    for name, metrics in evaluation.items():
        rows.append({"Classifier": name, **{key: metrics[key] for key in ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "average_precision"]}})
    performance = pd.DataFrame(rows).set_index("Classifier")
    st.dataframe(performance.style.format("{:.1%}"), width="stretch")
    st.subheader("Metric comparison")
    st.bar_chart(performance[["accuracy", "precision", "recall", "f1", "specificity"]])
    st.warning("The positive class is only 5.2% of this synthetic dataset. Accuracy alone is misleading; inspect recall, precision, average precision, and confusion matrices too.")


elif selected_page == "Confusion Matrix":
    st.title("Confusion matrix", icon=":material/grid_on:")
    model_name = st.selectbox("Classifier", list(models))
    metrics = evaluation[model_name]
    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    matrix = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: No asthma (0)", "Actual: Asthma (1)"], columns=["Predicted: No asthma (0)", "Predicted: Asthma (1)"])
    st.dataframe(matrix, width="stretch")
    st.dataframe(metric_table(metrics).style.format({"Value": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption("False negatives are asthma-labeled records predicted as no asthma. False positives are no-asthma records predicted as asthma.")


elif selected_page == "Feature Analysis":
    st.title("Feature analysis", icon=":material/insights:")
    tree_pipeline = models["Decision Tree"]
    preprocessor = tree_pipeline.named_steps["preprocess"]
    classifier = tree_pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({"Encoded feature": names, "Importance": classifier.feature_importances_})
    importance = importance.sort_values("Importance", ascending=False).head(15).set_index("Encoded feature")
    st.write("The chart shows the Decision Tree's impurity-based feature importance for this one trained split. It describes the model, not causal effects.")
    st.bar_chart(importance)
    st.dataframe(importance.reset_index().style.format({"Importance": "{:.3f}"}), hide_index=True, width="stretch")


else:
    st.title("About this project", icon=":material/info:")
    st.write("This academic project demonstrates a complete pattern-recognition workflow: dataset loading, preprocessing, supervised training, holdout evaluation, and interactive prediction.")
    st.markdown("**Dataset:** [Asthma Disease Dataset on Kaggle](https://doi.org/10.34740/KAGGLE/DSV/8669080), by Rabie El Kharoua (2024), CC BY 4.0.")
    st.write(metadata["limitations"])
    st.subheader("Interpretation")
    st.markdown("- **0:** the source dataset labels the record no asthma.\n- **1:** the source dataset labels the record asthma.\n- **Risk level:** a display band over an uncalibrated model score; it is not a clinical probability.")
    st.info("Input values outside the observed training range may produce unreliable results. Gender, ethnicity, and education appear as source codes because the dataset does not define user-facing category names.")
