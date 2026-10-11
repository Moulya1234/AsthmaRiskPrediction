# Asthma risk recognition demo

This project trains four classifiers on the original Kaggle Asthma Disease Dataset and provides a local Streamlit app with a dashboard, person-level prediction form, dataset analysis, algorithm descriptions, model comparison, confusion matrices, feature analysis, and project information.

## Important limitations

The dataset is synthetic and has 2,392 rows, of which 124 (5.2%) have `Diagnosis = 1`. The models use class weighting where supported so they can return positive labels, but this cannot make synthetic labels representative of real patients. In the current 80/20 holdout run, the best model had 56% asthma recall and 7% precision. A High / Moderate / Low band is an **uncalibrated model-score band**, not a validated medical risk estimate. Do not use the result to diagnose, treat, or make health decisions.

## Labels and risk bands

- **Dataset label 0:** the source record is labeled no asthma.
- **Dataset label 1:** the source record is labeled asthma.
- **Low model-score band:** score below 35%.
- **Moderate model-score band:** score from 35% to below 65%.
- **High model-score band:** score of 65% or higher.

The model predicts label 1 at a score of 50% or higher. Because the dataset is highly imbalanced and synthetic, the score must not be interpreted as the actual probability that a person has asthma.

## Data source

Rabie El Kharoua (2024), [Asthma Disease Dataset](https://doi.org/10.34740/KAGGLE/DSV/8669080), Kaggle, CC BY 4.0. The source describes the dataset as synthetic. The project excludes `PatientID` and `DoctorInCharge` from model features and treats `Gender`, `Ethnicity`, and `EducationLevel` as categorical codes because the dataset does not provide verified labels for those codes.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python train_model.py
python -m streamlit run app.py
```

If activation is blocked, use `\.venv\Scripts\python.exe` in place of `python`. KaggleHub may ask for Kaggle credentials if this dataset is not already cached. Alternatively, download the CSV from Kaggle and place it at `data/asthma_disease_data.csv`.

## Notebook

Run `jupyter notebook` and open `train_model.ipynb`. The same notebook can be uploaded to Colab along with the scripts and dataset. It runs the same downloader and training script as the local workflow.

## Project files

```text
app.py                    Multi-page app and person-level prediction form
download_dataset.py       Retrieves the Kaggle dataset
train_model.py            Preprocessing, four classifiers, evaluation and export
train_model.ipynb         Notebook workflow
data/asthma_disease_data.csv
models/asthma_models.joblib
models/asthma_model_metadata.json
```
