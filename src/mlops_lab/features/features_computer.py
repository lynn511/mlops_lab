import pandas as pd
from .base_features_computer import BaseFeaturesComputer

class TitanicFeaturesComputer(BaseFeaturesComputer):
    """Feature engineering class for Titanic dataset."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._extract_title(df)
        df = self._create_family_size(df)
        df = self._create_fare_group(df)
        df = self._drop_unused_columns(df)
        return df

    def _extract_title(self, df):
        df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        df['Title'] = df['Title'].replace(
            ['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
            'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        return df

    def _create_family_size(self, df):
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

    def _create_fare_group(self, df):
        df['FareGroup'] = pd.cut(
            df['Fare'],
            bins=[0, 10, 50, 100, 600],
            labels=['Low', 'Mid', 'High', 'Very High']
        )
        return df

    def _drop_unused_columns(self, df):
        return df.drop(columns=['Name','Parch','SibSp','Ticket','PassengerId'], errors="ignore")
