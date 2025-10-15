import argparse
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def main(input_csv, output_model):
    # Load engineered features
    df = pd.read_csv(input_csv)

    # Separate features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Initialize model
    model = LogisticRegression(max_iter=500)

    # Train model
    model.fit(X, y)

    # Save model to file
    with open(output_model, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression Model")
    parser.add_argument("--input", required=True, help="Path to features CSV")
    parser.add_argument("--output", required=True, help="Path to save trained model")
    args = parser.parse_args()

    main(args.input, args.output)
