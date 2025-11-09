"""
Feature engineering script for Titanic survival prediction.
Handles feature creation, splitting, and transformer building using package classes.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your package modules
from mlops_lab.preprocessing.dataloader import DataLoader
from mlops_lab.features.features_computer import TitanicFeaturesComputer
from mlops_lab.features.transformer_builder import TransformerBuilder


def main():
    parser = argparse.ArgumentParser(description="Apply feature engineering to Titanic dataset")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output_train", type=str, required=True, help="Output path for processed training data")
    parser.add_argument("--output_test", type=str, required=True, help="Output path for processed test data")
    parser.add_argument("--transformer_dir", type=str, default="transformers", help="Directory to save fitted transformers")
    parser.add_argument("--eval_dir", type=str, default="data/eval", help="Directory to save evaluation data")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory to save training split")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of train data for evaluation")
    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transformer_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    Path(args.train_dir).mkdir(parents=True, exist_ok=True)

    # 1️⃣ Load raw data via DataLoader
    print("📥 Loading data...")
    loader = DataLoader()
    train, test = loader.load(args.train_path, args.test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    # 2️⃣ Compute features via TitanicFeaturesComputer
    print("🧠 Applying feature engineering to train data...")
    fc = TitanicFeaturesComputer()
    train_features = fc.compute(train)

    # 3️⃣ Split into X_train / X_eval / y_train / y_eval
    print("✂️ Splitting train data into training and evaluation sets...")
    X = train_features.drop(columns=["Survived"])
    y = train_features["Survived"]
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Save evaluation split
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    X_eval.to_csv(eval_dir / "X_eval.csv", index=False)
    y_eval.to_csv(eval_dir / "y_eval.csv", index=False)
    print(f"📄 Evaluation data saved to: {eval_dir/'X_eval.csv'} and {eval_dir/'y_eval.csv'}")

    # Save training split
    train_dir = Path(args.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(train_dir / "X_train.csv", index=False)
    y_train.to_csv(train_dir / "y_train.csv", index=False)
    print(f"📄 Training split saved to: {train_dir/'X_train.csv'} and {train_dir/'y_train.csv'}")

    # Recombine X_train + y_train for saving processed train file
    train_df = X_train.copy()
    train_df["Survived"] = y_train
    train_df.to_csv(args.output_train, index=False)

    # 4️⃣ Featurize test set
    print("🧠 Applying feature engineering to test data...")
    test_features = fc.compute(test)
    test_features.to_csv(args.output_test, index=False)

    print(f"✅ Train features saved to: {args.output_train}")
    print(f"✅ Test features saved to: {args.output_test}")
    print(f"Final train shape: {train_df.shape}, test shape: {test_features.shape}")

    # 5️⃣ Build & fit transformers
    print("🏗 Building and fitting transformers on training data only...")
    tm = TransformerBuilder(output_dir=args.transformer_dir)
    num_cat_trans, bins = tm.build()
    tm.fit_and_save(X_train, num_cat_trans, bins)

    print(f"🎉 Featurization completed successfully! Transformers saved to: {args.transformer_dir}")


if __name__ == "__main__":
    main()


"""python scripts/featurize.py \
  --train_path data/processed/train_processed.csv \
  --test_path data/processed/test_processed.csv \
  --output_train data/featurized/train_features.csv \
  --output_test data/featurized/test_features.csv \
  --transformer_dir data/transformers \
  --eval_dir data/eval \
  --train_dir data/train \
  --test_size 0.2
"""