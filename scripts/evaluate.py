# evaluate.py
import argparse
import pickle
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_model(model_path: Path):
    """Load trained pipeline from pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_eval_data(X_path: Path, y_path: Path):
    """Load evaluation features and labels from separate CSV files."""
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()  # convert single-column DF to Series
    return X, y


def evaluate(model, X, y):
    """Compute evaluation metrics."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    clf_report = classification_report(y, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y, y_pred).tolist()
    return {
        "accuracy": acc,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat,
        "n_samples": int(len(y))
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on an evaluation dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (pickle)")
    parser.add_argument("--X_path", type=str, required=True, help="Path to evaluation features CSV (X_val.csv)")
    parser.add_argument("--y_path", type=str, required=True, help="Path to evaluation labels CSV (y_val.csv)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    X_path = Path(args.X_path)
    y_path = Path(args.y_path)

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    print(f"Loading evaluation data from: {X_path} and {y_path}")
    X_eval, y_eval = load_eval_data(X_path, y_path)

    print("Running evaluation...")
    metrics = evaluate(model, X_eval, y_eval)

    # Print summary
    print(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
    print("Classification report:\n")
    print(classification_report(y_eval, model.predict(X_eval)))

    print("Confusion matrix:")
    print(pd.DataFrame(metrics["confusion_matrix"], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))
    print(f"\nNumber of evaluation samples: {metrics['n_samples']}")

    # Optionally save metrics as JSON
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()

"""python scripts/evaluate.py \
  --model_path models/random_forest_pipeline.pkl \
  --X_path data/eval/X_eval.csv \
  --y_path data/eval/y_eval.csv \
  --output_json reports/metrics.json
"""