"""
Preprocessor class for Titanic dataset.
Handles loading, cleaning, and splitting data.
"""

import pandas as pd
from pathlib import Path


class Preprocessor:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.train = None
        self.test = None
        self.df = None

    def load_data(self):
        """Load training and test CSVs."""
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        return self.train, self.test

    def clean_data(self):
        """Clean and combine train/test data."""
        train, test = self.train.copy(), self.test.copy()
        train.drop(columns=["Cabin"], inplace=True, errors="ignore")
        test.drop(columns=["Cabin"], inplace=True, errors="ignore")

        train["Embarked"].fillna("S", inplace=True)
        test["Fare"].fillna(test["Fare"].mean(), inplace=True)

        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
        self.df = df
        return df

    def split_data(self):
        """Split unified dataframe back into train/test sets."""
        df = self.df.copy()
        train = df.loc[:890].copy()
        test = df.loc[891:].copy()

        if "Survived" in test.columns:
            test.drop(columns=["Survived"], inplace=True)
        if "Survived" in train.columns:
            train["Survived"] = train["Survived"].astype("int64")

        self.train, self.test = train, test
        return train, test
