from typing import Tuple
import pandas as pd

class Splitting:
    """
    Split the unified dataframe back into train and test sets.
    This mirrors your original split_data function.
    """

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Keep the same slicing used before: rows 0..890 -> train, 891..end -> test
        train = df.loc[:890].copy()
        test = df.loc[891:].copy()

        if "Survived" in test.columns:
            test.drop(columns=["Survived"], inplace=True)

        if "Survived" in train.columns:
            try:
                train["Survived"] = train["Survived"].astype("int64")
            except Exception:
                pass

        return train, test
