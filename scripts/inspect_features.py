# scripts/inspect_features.py
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_model(model_path):
    """Load trained pipeline from pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from: {model_path}")
    return model

def get_feature_names_from_pipeline(pipeline):
    """Extract feature names after ColumnTransformer + OneHotEncoding."""
    transformer = pipeline.named_steps['num_cat_transformation']
    feature_names = []

    for name, trans, cols in transformer.transformers_:
        if name == 'remainder' and trans == 'drop':
            continue
        if hasattr(trans, 'get_feature_names_out'):
            trans_feature_names = trans.get_feature_names_out(trans.feature_names_in_)
            feature_names.extend([f"{name}__{f}" for f in trans_feature_names])
        else:
            # Fallback for scalers and encoders without get_feature_names_out
            # If no feature_names_in_ attribute, use original cols
            if hasattr(trans, 'feature_names_in_'):
                col_names = trans.feature_names_in_
            else:
                col_names = cols
            feature_names.extend([f"{name}__{c}" for c in col_names])

    return feature_names


def plot_feature_importances(feature_names, importances):
    """Plot feature importances sorted descending."""
    idx_sorted = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx_sorted]
    sorted_importances = importances[idx_sorted]

    plt.figure(figsize=(10, 8))
    plt.barh(sorted_names, sorted_importances)
    plt.gca().invert_yaxis()
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Inspect feature importances of trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (pickle)")
    args = parser.parse_args()

    model = load_model(args.model_path)
    feature_names = get_feature_names_from_pipeline(model)

    if hasattr(model.named_steps['classifier'], "feature_importances_"):
        importances = model.named_steps['classifier'].feature_importances_
    else:
        raise ValueError("This model does not have feature_importances_. Try a tree-based model like RandomForest.")

    # Display in console
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    print("\n📊 Feature Importances:")
    print(importance_df)

    # Plot
    plot_feature_importances(feature_names, importances)

if __name__ == "__main__":
    main()
