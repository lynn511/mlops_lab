import argparse
from mlops_lab.models.random_forest import RandomForestModel
from mlops_lab.models.xgboost import XGBoostModel
from mlops_lab.models.logistic_reg import LogisticRegressionModel
from pathlib import Path
import pickle
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest model for Titanic dataset")
    parser.add_argument("--x_train_path", type=str, default="data/train/X_train.csv")
    parser.add_argument("--y_train_path", type=str, default="data/train/y_train.csv")
    parser.add_argument("--transformer_dir", type=str, default="data/transformers")
    parser.add_argument("--output_model", type=str, default="models/random_forest_pipeline.pkl")
    args = parser.parse_args()

    print("📥 Loading training data...")
    X_train = pd.read_csv(args.x_train_path)
    y_train = pd.read_csv(args.y_train_path).squeeze()

    print("⚙️ Loading transformers...")
    with open(Path(args.transformer_dir) / "num_cat_transformer.pkl", "rb") as f:
        num_cat_trans = pickle.load(f)
    with open(Path(args.transformer_dir) / "bins_transformer.pkl", "rb") as f:
        bins_trans = pickle.load(f)

    print("🚀 Initializing Model...")
    #model = RandomForestModel(num_cat_trans, bins_trans, output_path=args.output_model)
    #model = XGBoostModel(num_cat_trans, bins_trans, args.output_model)
    model = LogisticRegressionModel(num_cat_trans, bins_trans, args.output_model)

    print("🎯 Training model...")
    model.train(X_train, y_train)

    print("✅ Training complete and model saved!")


if __name__ == "__main__":
    main()


"""uv run python scripts/train.py \
  --train_dir data/train \
  --transformer_dir data/transformers \
  --output_model models/random_forest_pipeline.pkl
"""