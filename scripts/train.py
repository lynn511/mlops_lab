import pandas as pd
import pickle
import argparse
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Data
# -------------------------------
def load_data(x_train_path, y_train_path):
    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path).squeeze()  # squeeze() converts single column DF to Series
    if y.name != 'Survived':
        y.name = 'Survived'
    return X, y

# -------------------------------
# Load Saved Transformers
# -------------------------------
def load_transformers(transformer_dir):
    num_cat_path = Path(transformer_dir) / "num_cat_transformer.pkl"
    bins_path = Path(transformer_dir) / "bins_transformer.pkl"
    with open(num_cat_path, 'rb') as f:
        num_cat_trans = pickle.load(f)
    with open(bins_path, 'rb') as f:
        bins_trans = pickle.load(f)
    return num_cat_trans, bins_trans

# -------------------------------
# Create Pipeline
# -------------------------------
def create_pipeline(algo, num_cat_trans, bins_trans):
    return Pipeline([
        ('num_cat_transformation', num_cat_trans),
        ('bins', bins_trans),
        ('classifier', algo)
    ])

# -------------------------------
# Save Pipeline
# -------------------------------
def save_pipeline(pipeline, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"✅ Pipeline saved to: {output_path}")

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Titanic model with saved transformers")
    parser.add_argument("--x_train_path", type=str, default="data/train/X_train.csv", help="Path to X_train CSV file")
    parser.add_argument("--y_train_path", type=str, default="data/train/y_train.csv", help="Path to y_train CSV file")
    parser.add_argument("--transformer_dir", type=str, default="data/transformers", help="Directory where transformers are saved")
    parser.add_argument("--output_model", type=str, default="models/linear_svc_pipeline.pkl", help="Path to save trained pipeline")
    args = parser.parse_args()

    print("📥 Loading training data...")
    X_train, y_train = load_data(args.x_train_path, args.y_train_path)
    print(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")

    print("⚙️ Loading saved transformers...")
    num_cat_trans, bins_trans = load_transformers(args.transformer_dir)

    # 👇 You can toggle between models here
    #svc_model = LinearSVC(max_iter=10000, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    pipeline = create_pipeline(rf_model, num_cat_trans, bins_trans)

    print("🚀 Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    print("💾 Saving pipeline...")
    save_pipeline(pipeline, args.output_model)

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()

"""python scripts/train.py \    
  --x_train_path data/train/X_train.csv \
  --y_train_path data/train/y_train.csv \
  --transformer_dir data/transformers \
  --output_model models/random_forest_pipeline.pkl"""