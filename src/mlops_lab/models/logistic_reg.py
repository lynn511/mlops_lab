from mlops_lab.models.base_model import BaseModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model with preprocessing transformers."""

    def __init__(self, num_cat_trans, bins_trans, output_path="models/logistic_pipeline.pkl"):
        self.num_cat_trans = num_cat_trans
        self.bins_trans = bins_trans
        self.output_path = Path(output_path)
        self.pipeline = None

    def train(self, X, y):
        """Train Logistic Regression model inside a pipeline."""
        print("🚀 Training Logistic Regression model...")
        self.pipeline = Pipeline([
            ('num_cat_transformation', self.num_cat_trans),
            ('bins', self.bins_trans),
            ('classifier', LogisticRegressionCV(cv=5, max_iter=1000))
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
        """Evaluate the model accuracy."""
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"📊 Accuracy: {acc:.4f}")
        return acc
