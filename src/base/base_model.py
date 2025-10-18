"""
Base Model abstract class.

Provides a common interface for all machine learning models.

Responsibilities:
- ML Engineer: Primary owner - implements concrete model classes
- Data Scientist: Uses models for experimentation and hyperparameter tuning
- Software Engineer: Maintains base class and ensures SOLID principles
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.
    
    This class provides a consistent interface for training, predicting,
    evaluating, and saving/loading models.
    
    Attributes:
        model: The underlying ML model instance
        model_name (str): Name of the model
        config (Dict[str, Any]): Model configuration
        is_fitted (bool): Whether the model has been trained
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Name identifier for the model
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self._training_metrics = {}
        logger.info(f"Initialized {self.__class__.__name__}: {model_name}")
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the underlying ML model instance.
        
        Returns:
            Any: The model instance (e.g., sklearn estimator, XGBoost model)
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            
        Returns:
            self: Fitted model instance for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predictions
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Generate probability predictions (for classification models).
        
        Args:
            X: Features to predict on
            
        Returns:
            Optional[np.ndarray]: Probability predictions, None if not applicable
        """
        self._check_is_fitted()
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        logger.warning(f"{self.model_name} does not support probability predictions")
        return None
    
    @abstractmethod
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            metrics: Optional list of metric names to compute
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        pass
    
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        self._check_is_fitted()
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'config': self.config,
            'training_metrics': self._training_metrics,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, path: Path) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            BaseModel: Loaded model instance
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model_data = joblib.load(load_path)
        
        # Reconstruct the model instance
        instance = cls(
            model_name=model_data['model_name'],
            config=model_data['config']
        )
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance._training_metrics = model_data.get('training_metrics', {})
        
        logger.info(f"Model loaded from {load_path}")
        return instance
    
    def _check_is_fitted(self) -> None:
        """
        Check if the model has been fitted.
        
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError(
                f"{self.model_name} must be fitted before making predictions. "
                "Call fit() first."
            )
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Model configuration and hyperparameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Key-value pairs of parameters to set
            
        Returns:
            self: Model instance for method chaining
        """
        self.config.update(params)
        if self.model and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    def get_training_metrics(self) -> Dict[str, float]:
        """
        Get metrics from the last training session.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        return self._training_metrics.copy()
