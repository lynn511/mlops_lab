import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn import set_config

set_config(display='diagram')


def load_data(path):
    df = pd.read_csv(path)
    return df


def create_feature_pipeline():
    num_cat_transformation = ColumnTransformer([
        ('scaling', MinMaxScaler(), [0, 2]),
        ('onehotencolding1', OneHotEncoder(), [1, 3]),
        ('ordinal', OrdinalEncoder(), [4]),
        ('onehotencolding2', OneHotEncoder(), [5, 6])
    ], remainder='passthrough')

    bins = ColumnTransformer([
        ('Kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2])
    ], remainder='passthrough')

    def pipeline_creator(algo):
        return Pipeline([
            ('num_cat_transformation', num_cat_transformation),
            ('bins', bins),
            ('classifier', algo)
        ])

    return pipeline_creator


def main(input_csv, output_csv):
    df = load_data(input_csv)

    # Separate train/test like in the notebook
    train = df.loc[:890].copy()
    test = df.loc[891:].copy()

    test.drop(columns=['Survived'], inplace=True)
    train['Survived'] = train['Survived'].astype('int64')

    train = train.drop("PassengerId", axis=1)

    X_train = train.drop("Survived", axis=1)
    y_train = train["Survived"]

    # Save features (X_train + y_train) to CSV for later use
    X_train['Survived'] = y_train
    X_train.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic Feature Engineering Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV (combined or train)")
    parser.add_argument("--output", required=True, help="Output CSV with features")
    args = parser.parse_args()

    main(args.input, args.output)