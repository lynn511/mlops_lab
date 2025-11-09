from abc import ABC, abstractmethod
import pandas as pd

class BaseFeaturesComputer(ABC):
    """Abstract base class for feature engineering."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a DataFrame and returns one with engineered features."""
        pass
