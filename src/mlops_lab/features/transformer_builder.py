import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer


class TransformerBuilder:
    """Builds, fits, and saves feature transformation pipelines."""

    def __init__(self, output_dir="transformers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self):
        num_cat_transformation = ColumnTransformer([
            ('scaling', MinMaxScaler(), [0, 2]),
            ('onehotencoding1', OneHotEncoder(), [1, 3]),
            ('ordinal', OrdinalEncoder(), [4]),
            ('onehotencoding2', OneHotEncoder(), [5, 6, 7])
        ], remainder='passthrough')

        bins = ColumnTransformer([
            ('kbins', KBinsDiscretizer(
                n_bins=10,
                encode='ordinal',
                strategy='quantile',
                quantile_method='averaged_inverted_cdf'
            ), [0, 2])
        ], remainder='passthrough')

        return num_cat_transformation, bins

    def fit_and_save(self, X_train, num_cat_transformation, bins):
        num_cat_transformation.fit(X_train)
        bins.fit(X_train)

        with open(self.output_dir / "num_cat_transformer.pkl", "wb") as f:
            pickle.dump(num_cat_transformation, f)

        with open(self.output_dir / "bins_transformer.pkl", "wb") as f:
            pickle.dump(bins, f)

        print(f"✅ Transformers saved to '{self.output_dir}' directory.")

