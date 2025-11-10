from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Evaluate the model."""
        pass
