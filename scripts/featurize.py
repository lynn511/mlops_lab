import pandas as pd
import argparse
from pathlib import Path
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split

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

def create_fare_group(df):
    df['FareGroup'] = pd.cut(
        df['Fare'],
        bins=[0, 10, 50, 100, 600],
        labels=['Low', 'Mid', 'High', 'Very High']
    )
    return df


def drop_unused_columns(df):
    df.drop(columns=['Name','Parch','SibSp','Ticket','PassengerId'], inplace=True)
    return df

# -------------------------------
# Feature Transformation Pipelines
# -------------------------------

def build_feature_pipelines():
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


def fit_and_save_transformers(X_train, num_cat_transformation, bins, output_dir="transformers"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_cat_transformation.fit(X_train)
    bins.fit(X_train)
    with open(Path(output_dir)/"num_cat_transformer.pkl", "wb") as f:
        pickle.dump(num_cat_transformation, f)
    with open(Path(output_dir)/"bins_transformer.pkl", "wb") as f:
        pickle.dump(bins, f)
    print(f"✅ Transformers saved to '{output_dir}' directory.")

# -------------------------------
# Master function to apply all features
# -------------------------------

def apply_feature_engineering(df):
    df = extract_title(df)
    df = create_family_size(df)
    df = create_fare_group(df)   # 👈 added
    df = drop_unused_columns(df)
    return df

# -------------------------------
# Split train/eval after feature engineering
# -------------------------------

def split_train_eval(df, eval_dir="data/eval", test_size=0.2, random_state=42, train_dir="data/train"):
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save evaluation set
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    X_eval.to_csv(eval_dir/"X_eval.csv", index=False)
    y_eval.to_csv(eval_dir/"y_eval.csv", index=False)
    print(f"📄 Evaluation data saved to: {eval_dir/'X_eval.csv'} and {eval_dir/'y_eval.csv'}")

    # Save training split
    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(train_dir/"X_train.csv", index=False)
    y_train.to_csv(train_dir/"y_train.csv", index=False)
    print(f"📄 Training split saved to: {train_dir/'X_train.csv'} and {train_dir/'y_train.csv'}")

    # Return full train dataframe for transformer fitting
    train_df = X_train.copy()
    train_df['Survived'] = y_train
    return train_df

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
    parser.add_argument("--eval_dir", type=str, default="data/eval", help="Directory to save evaluation data")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory to save training split")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of train data for evaluation")
    args = parser.parse_args()

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transformer_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    Path(args.train_dir).mkdir(parents=True, exist_ok=True)

    print("📥 Loading data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    print("🧠 Applying feature engineering to train data...")
    train_features = apply_feature_engineering(train)

    print("✂️ Splitting train data into training and evaluation sets...")
    train_features = split_train_eval(
        train_features, eval_dir=args.eval_dir, test_size=args.test_size, train_dir=args.train_dir
    )

    print("🧠 Applying feature engineering to test data...")
    test_features = apply_feature_engineering(test)

    print("💾 Saving feature-engineered data...")
    train_features.to_csv(args.output_train, index=False)
    test_features.to_csv(args.output_test, index=False)

    print(f"✅ Train features saved to: {args.output_train}")
    print(f"✅ Test features saved to: {args.output_test}")
    print(f"Final train shape: {train_features.shape}, test shape: {test_features.shape}")

    print("🏗 Building and fitting transformers on training data only...")
    train_features_for_fit = train_features.drop(columns=['Survived'], errors='ignore')
    num_cat_trans, bins = build_feature_pipelines()
    fit_and_save_transformers(train_features_for_fit, num_cat_trans, bins, output_dir=args.transformer_dir)

    print("🎉 Featurization completed successfully!")

if __name__ == "__main__":
    main()


"""python scripts/featurize.py \
  --train_path data/processed/train_processed.csv \
  --test_path data/processed/test_processed.csv \
  --output_train data/featurized/train_features.csv \
  --output_test data/featurized/test_features.csv \
  --transformer_dir data/transformers \
  --eval_dir data/eval \
  --train_dir data/train \
  --test_size 0.2
"""