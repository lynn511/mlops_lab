from abc import ABC, abstractmethod
import pandas as pd

class BasePreprocessor(ABC):
    """Abstract base class for any preprocessing step."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a DataFrame and returns a transformed DataFrame."""
        pass
