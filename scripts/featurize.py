import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_data(path):
    return pd.read_csv(path)


def create_feature_pipeline():
    transformer = ColumnTransformer([
        ('scale', MinMaxScaler(), ['Age', 'Fare', 'Parch', 'SibSp']),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Embarked', 'Pclass', 'Sex'])
    ], remainder='drop')  # drop Name, Ticket, PassengerId automatically
    return transformer


def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Separate target
    y = df['Survived']
    X = df.drop(columns=['Survived'])

    # Create transformer and apply it
    pipeline = create_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)

    # Convert to DataFrame and reattach target
    X_out = pd.DataFrame(X_transformed)
    X_out['Survived'] = y.reset_index(drop=True)

    X_out.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic Feature Engineering")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.input, args.output)
