import argparse
import pickle
from pathlib import Path
import pandas as pd

# -------------------------------
# Load Model
# -------------------------------
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from: {model_path}")
    return model

# -------------------------------
# Load Features
# -------------------------------
def load_features(features_path):
    X = pd.read_csv(features_path)
    print(f"📊 Loaded features with shape: {X.shape}")
    return X

# -------------------------------
# Predict
# -------------------------------
def make_predictions(model, X):
    y_pred = model.predict(X)
    print(f"🪄 Generated {len(y_pred)} predictions.")
    return y_pred

# -------------------------------
# Save Predictions
# -------------------------------
def save_predictions(predictions, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Prediction": predictions}).to_csv(output_path, index=False)
    print(f"💾 Predictions saved to: {output_path}")

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (pickle)")
    parser.add_argument("--features_path", type=str, required=True, help="Path to CSV with features (no labels)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions CSV")
    args = parser.parse_args()

    model = load_model(args.model_path)
    X = load_features(args.features_path)
    preds = make_predictions(model, X)
    save_predictions(preds, args.output_path)

    print("✅ Inference complete!")

if __name__ == "__main__":
    main()


"""python scripts/predict.py \
  --model_path models/random_forest_pipeline.pkl \
  --features_path data/featurized/test_features.csv \
  --output_path data/predictions/test_predictions.csv
"""