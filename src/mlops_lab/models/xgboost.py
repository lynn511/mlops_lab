from mlops_lab.models.base_model import BaseModel
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

# XGBoost scikit-learn API
from xgboost import XGBClassifier

class XGBoostModel(BaseModel):
    def __init__(self, num_cat_trans, bins_trans, output_path):
        self.num_cat_trans = num_cat_trans
        self.bins_trans = bins_trans
        self.output_path = output_path
        self.pipeline = None

    def train(self, X, y):
        self.pipeline = Pipeline([
            ('num_cat_transformation', self.num_cat_trans),
            ('bins', self.bins_trans),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
        self.pipeline.fit(X, y)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Pipeline saved to: {self.output_path}")

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)