# feature.py
import pandas as pd
import argparse
from pathlib import Path
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer

# -------------------------------
# Feature Creation Functions
# -------------------------------

def extract_title(df):
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df['Title'] = df['Title'].replace(
        ['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
         'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def create_family_size(df):
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    def family_size_bin(number):
        if number == 1:
            return "Alone"
        elif number < 5:
            return "Small"
        else:
            return "Large"
    df['Family_size'] = df['Family_size'].apply(family_size_bin)
    return df

def drop_unused_columns(df):
    df.drop(columns=['Name','Parch','SibSp','Ticket','PassengerId'], inplace=True)
    print(df.head())
    return df

# -------------------------------
# Feature Transformation Pipelines
# -------------------------------

def build_feature_pipelines():
    """
    Builds column transformation pipelines for numeric and categorical features.
    """
    num_cat_transformation = ColumnTransformer([
        ('scaling', MinMaxScaler(), [0, 2]),       # numeric columns
        ('onehot1', OneHotEncoder(), [1, 3]),      # categorical columns
        ('ordinal', OrdinalEncoder(), [4]),        # ordinal column
        ('onehot2', OneHotEncoder(), [5, 6])       # other categorical columns
    ], remainder='passthrough')

    bins = ColumnTransformer([
        ('kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2])
    ], remainder='passthrough')

    return num_cat_transformation, bins

def fit_and_save_transformers(X_train, num_cat_transformation, bins, output_dir="transformers"):
    """
    Fits the transformers on the training data and saves them as pickle files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Fit transformers
    num_cat_transformation.fit(X_train)
    bins.fit(X_train)

    # Save transformers
    with open(Path(output_dir)/"num_cat_transformer.pkl", "wb") as f:
        pickle.dump(num_cat_transformation, f)
    with open(Path(output_dir)/"bins_transformer.pkl", "wb") as f:
        pickle.dump(bins, f)

    print(f"Transformers saved to '{output_dir}' directory.")

# -------------------------------
# Master function to apply all features
# -------------------------------

def apply_feature_engineering(df):
    df = extract_title(df)
    df = create_family_size(df)
    df = drop_unused_columns(df)
    return df

# -------------------------------
# Main function
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply feature engineering to Titanic dataset")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output_train", type=str, required=True, help="Output path for processed training data")
    parser.add_argument("--output_test", type=str, required=True, help="Output path for processed test data")
    parser.add_argument("--transformer_dir", type=str, default="transformers", help="Directory to save fitted transformers")
    args = parser.parse_args()

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transformer_dir).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    print("Applying feature engineering to train data...")
    train_features = apply_feature_engineering(train)
    print("Applying feature engineering to test data...")
    test_features = apply_feature_engineering(test)

    print("Saving feature-engineered data...")
    train_features.to_csv(args.output_train, index=False)
    test_features.to_csv(args.output_test, index=False)

    print(f"Train features saved to: {args.output_train}")
    print(f"Test features saved to: {args.output_test}")
    print(f"Final train shape: {train_features.shape}")
    print(f"Final test shape: {test_features.shape}")

    # Build transformers and save
    print("Building transformers...")
    num_cat_trans, bins = build_feature_pipelines()
    train_features_for_fit = train_features.drop(columns=['Survived'], errors='ignore')
    fit_and_save_transformers(train_features_for_fit, num_cat_trans, bins, output_dir=args.transformer_dir)

if __name__ == "__main__":
    main()