
# scripts/evaluate.py
import argparse
from pathlib import Path
import json
import pandas as pd
import pickle

from mlops_lab.models.logistic_reg import LogisticRegressionModel
import sys
# ----------------------------
# Helper functions
# ----------------------------
def load_eval_data(X_path: Path, y_path: Path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    return X, y

def load_model(model_path: Path) -> LogisticRegressionModel:
    """Load the trained pipeline and wrap it in LogisticRegressionModel."""
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Create a dummy model instance and attach the loaded pipeline
    model = LogisticRegressionModel(num_cat_trans=None, bins_trans=None, output_path=model_path)
    model.pipeline = pipeline
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--X_path", type=str, required=True)
    parser.add_argument("--y_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    X_eval, y_eval = load_eval_data(Path(args.X_path), Path(args.y_path))

    print(f"Loading model from: {args.model_path}")
    model = load_model(Path(args.model_path))

    print("Running evaluation...")
    accuracy = model.evaluate(X_eval, y_eval)
    y_pred = model.predict(X_eval)

    from sklearn.metrics import classification_report, confusion_matrix

    metrics = {
        "accuracy": accuracy,
        "classification_report": classification_report(y_eval, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_eval, y_pred).tolist(),
        "n_samples": int(len(y_eval))
    }

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print("Classification report:\n", classification_report(y_eval, y_pred))
    print("Confusion matrix:\n", pd.DataFrame(metrics["confusion_matrix"], index=["true_0","true_1"], columns=["pred_0","pred_1"]))
    print(f"Number of evaluation samples: {metrics['n_samples']}")

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