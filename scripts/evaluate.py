import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def compute_metrics(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    return {"accuracy": float(acc)}


def main(model_path: str, data_path: str, metrics_output: str | None):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    print(f"Loading evaluation data from: {data_path}")
    df = load_data(data_path)

    if "Survived" not in df.columns:
        raise ValueError("Evaluation CSV must contain a 'Survived' column as the target.")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    print("Computing metrics...")
    metrics = compute_metrics(model, X, y)

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if metrics_output:
        out_path = Path(metrics_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a labeled CSV")
    parser.add_argument("--model", required=True, help="Path to saved model (pickle)")
    parser.add_argument("--input", required=True, help="Path to evaluation CSV with 'Survived' column")
    parser.add_argument("--metrics_output", required=False, help="Optional path to save metrics as JSON")

    args = parser.parse_args()

    main(args.model, args.input, args.metrics_output)
