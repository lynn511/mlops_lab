from pathlib import Path
import pandas as pd
from typing import Tuple

class DataLoader:
    """Load training and test CSV files and return DataFrames."""

    def load(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test

