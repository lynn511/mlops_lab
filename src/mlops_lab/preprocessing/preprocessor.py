"""
Preprocessing utilities for Titanic dataset.
Handles data loading, cleaning, and splitting logic.
"""

import pandas as pd


def load_data(train_path, test_path):
    """Load training and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def clean_data(train, test):
    """Clean the data by handling missing values and dropping unnecessary columns."""
    train.drop(columns=["Cabin"], inplace=True)
    test.drop(columns=["Cabin"], inplace=True)

    train["Embarked"].fillna("S", inplace=True)
    test["Fare"].fillna(test["Fare"].mean(), inplace=True)

    df = pd.concat([train, test], sort=True).reset_index(drop=True)
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    return df


def split_data(df):
    """Split the unified dataframe back into train and test sets."""
    train = df.loc[:890].copy()
    test = df.loc[891:].copy()

    if "Survived" in test.columns:
        test.drop(columns=["Survived"], inplace=True)

    if "Survived" in train.columns:
        train["Survived"] = train["Survived"].astype("int64")

    return train, test
