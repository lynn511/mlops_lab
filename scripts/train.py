import pandas as pd
import pickle
import argparse
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# -------------------------------
# Load Data
# -------------------------------
def load_data(train_path):
    train = pd.read_csv(train_path)
    if 'Survived' not in train.columns:
        raise ValueError("Training data must contain 'Survived' column as target")
    X = train.drop(columns=['Survived'])
    y = train['Survived']
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
    parser = argparse.ArgumentParser(description="Train Titanic model with LinearSVC and saved transformers")
    parser.add_argument("--train_path", type=str, required=True, help="Path to featurized training CSV file")
    parser.add_argument("--transformer_dir", type=str, required=True, help="Directory where transformers are saved")
    parser.add_argument("--output_model", type=str, default="models/linear_svc_pipeline.pkl", help="Path to save trained pipeline")
    args = parser.parse_args()

    print("📥 Loading training data...")
    X_train, y_train = load_data(args.train_path)
    print(f"Training data shape: {X_train.shape}")

    print("⚙️ Loading saved transformers...")
    num_cat_trans, bins_trans = load_transformers(args.transformer_dir)

    print("🤖 Creating pipeline with LinearSVC...")
    svc_model = LinearSVC(max_iter=10000, random_state=42)
    pipeline = create_pipeline(svc_model, num_cat_trans, bins_trans)

    print("🚀 Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    print("💾 Saving pipeline...")
    save_pipeline(pipeline, args.output_model)

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()

