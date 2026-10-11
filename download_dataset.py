"""Download the original Kaggle Asthma Disease Dataset."""

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "asthma_disease_data.csv"
DATASET_SLUG = "rabieelkharoua/asthma-disease-dataset"


def main() -> None:
    if DATASET_PATH.exists():
        print(f"Dataset already available: {DATASET_PATH}")
        return

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install dependencies first with `pip install -r requirements.txt`.") from exc

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset {DATASET_SLUG}...")
    downloaded_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    matches = list(downloaded_dir.rglob("asthma_disease_data.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find asthma_disease_data.csv under {downloaded_dir}")
    shutil.copy2(matches[0], DATASET_PATH)
    print(f"Saved dataset: {DATASET_PATH}")
    print("Source: https://doi.org/10.34740/KAGGLE/DSV/8669080")


if __name__ == "__main__":
    main()
