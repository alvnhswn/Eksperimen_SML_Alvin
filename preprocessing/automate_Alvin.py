import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess_data(input_path, output_path):
    logging.info("Memulai preprocessing...")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    # Load dataset
    df = pd.read_csv(input_path)

    # Drop customerID
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna(subset=["TotalCharges"])

    # Encode target
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Split feature-target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical = X.select_dtypes(include="object").columns
    numerical = X.select_dtypes(exclude="object").columns

    # One Hot Encoding
    X = pd.get_dummies(
        X,
        columns=categorical,
        drop_first=True
    )

    # Scaling
    scaler = StandardScaler()

    X[numerical] = scaler.fit_transform(
        X[numerical]
    )

    # Merge
    processed = X.copy()
    processed["Churn"] = y

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed.to_csv(output_path, index=False)

    logging.info("Preprocessing selesai.")
    logging.info(f"Dataset disimpan pada {output_path}")


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    INPUT_PATH = BASE_DIR / "raw_dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    OUTPUT_PATH = BASE_DIR / "preprocessing" / "telco_preprocessed.csv"

    preprocess_data(INPUT_PATH, OUTPUT_PATH)