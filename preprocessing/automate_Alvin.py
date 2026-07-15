import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    """
    Melakukan preprocessing dataset Telco Customer Churn.
    """

    # Load data
    df = pd.read_csv(input_path)

    # Drop customerID
    df = df.drop(columns=["customerID"])

    # Konversi TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Hapus missing value
    df = df.dropna(subset=["TotalCharges"])

    # Encode target
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Pisahkan fitur
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_columns = X.select_dtypes(include="object").columns

    numerical_columns = X.select_dtypes(exclude="object").columns

    # One Hot Encoding
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

    # Standard Scaling
    scaler = StandardScaler()

    X[numerical_columns] = scaler.fit_transform(
        X[numerical_columns]
    )

    # Gabungkan kembali
    processed = X.copy()

    processed["Churn"] = y

    # Simpan
    processed.to_csv(output_path, index=False)

    print("Preprocessing selesai.")
    print(f"Dataset disimpan di {output_path}")


if __name__ == "__main__":

    preprocess_data(
        "../raw_dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_preprocessed.csv"
    )