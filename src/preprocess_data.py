import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"


CLASS_SPEC = [
    {
        "name": "pointing",
        "label": 0,
        "preferred_filenames": ["position1_pointing.csv", "pointing.csv"],
        "keywords": ["point", "pointing"],
    },
    {
        "name": "thumbs_up",
        "label": 1,
        "preferred_filenames": ["position2._thumbs_up.csv", "position2_thumbs_up.csv", "thumbs_up.csv"],
        "keywords": ["thumb", "thumbs"],
    },
    {
        "name": "okay_sign",
        "label": 2,
        "preferred_filenames": [
            "manus_hand_data_reset.csv",
            "position3_okay_sign.csv",
            "okay_sign.csv",
            "ok_sign.csv",
            "reset.csv",
        ],
        "keywords": ["reset", "okay", "ok_sign", "ok"],
    },
]


def resolve_dataset_file(spec):
    for filename in spec["preferred_filenames"]:
        candidate = RAW_DIR / filename
        if candidate.exists():
            return candidate

    csv_files = list(RAW_DIR.glob("*.csv"))
    for csv_file in csv_files:
        lowered = csv_file.name.lower()
        if any(keyword in lowered for keyword in spec["keywords"]):
            return csv_file

    expected = ", ".join(spec["preferred_filenames"])
    raise FileNotFoundError(
        f"Missing CSV for class '{spec['name']}'. Put one of [{expected}] in {RAW_DIR} "
        f"or include one of keywords {spec['keywords']} in the filename."
    )


def load_labeled_dataframe(spec):
    file_path = resolve_dataset_file(spec)
    df = pd.read_csv(file_path)
    df["label"] = spec["label"]
    print(f"Loaded {spec['name']:>10} ({spec['label']}): {file_path.name} -> {len(df)} rows")
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    class_frames = [load_labeled_dataframe(spec) for spec in CLASS_SPEC]
    combined_df = pd.concat(class_frames, ignore_index=True)

    features = combined_df.drop(columns=["label"])
    labels = combined_df["label"]

    # 70/15/15 split with stratification.
    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_scaled, columns=features.columns)
    val_df = pd.DataFrame(X_val_scaled, columns=features.columns)
    test_df = pd.DataFrame(X_test_scaled, columns=features.columns)

    train_df["label"] = y_train.values
    val_df["label"] = y_val.values
    test_df["label"] = y_test.values

    train_df.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val_data.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_data.csv", index=False)

    joblib.dump(scaler, ARTIFACTS_DIR / "minmax_scaler.pkl")

    class_mapping = {spec["label"]: spec["name"] for spec in CLASS_SPEC}
    with open(ARTIFACTS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    print("Saved: data/processed/{train_data.csv,val_data.csv,test_data.csv}")
    print("Saved: artifacts/minmax_scaler.pkl")
    print("Saved: artifacts/class_mapping.json")


if __name__ == "__main__":
    main()



