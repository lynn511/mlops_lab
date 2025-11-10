from pathlib import Path
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlops_lab.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model with preprocessing transformers."""

    def __init__(self, num_cat_trans, bins_trans, output_path="models/random_forest_pipeline.pkl"):
        self.num_cat_trans = num_cat_trans
        self.bins_trans = bins_trans
        self.output_path = Path(output_path)
        self.pipeline = None

    def train(self, X, y):
        """Train Random Forest model inside a pipeline."""
        print("🚀 Training Random Forest model...")
        self.pipeline = Pipeline([
            ('num_cat_transformation', self.num_cat_trans),
            ('bins', self.bins_trans),
            ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
        ])
        self.pipeline.fit(X, y)
        self.save()
        print("✅ Model trained and saved successfully!")

    def save(self):
        """Save the trained pipeline."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"💾 Pipeline saved to {self.output_path}")

    def predict(self, X):
        """Predict using the trained model."""
        if not self.pipeline:
            raise ValueError("Model not trained or loaded.")
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model and return accuracy."""
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"📊 Accuracy: {acc:.4f}")
        return acc
