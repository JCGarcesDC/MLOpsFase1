"""
Base Preprocessor abstract class.

Provides a common interface for all data preprocessing operations.

Responsibilities:
- Data Engineer: Primary owner - implements ETL preprocessing steps
- ML Engineer: Implements feature engineering and scaling steps
- Software Engineer: Maintains base class architecture
- Data Scientist: Uses concrete implementations for data transformation
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessing operations.
    
    This class provides a consistent interface for all preprocessing steps
    including cleaning, transformation, and feature engineering.
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for preprocessing
        is_fitted (bool): Whether the preprocessor has been fitted
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base preprocessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        self._fitted_params = {}
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'BasePreprocessor':
        """
        Fit the preprocessor to the training data.
        
        Learn any necessary parameters from the training data
        (e.g., means, medians, categories) without transforming it.
        
        Args:
            df: Input DataFrame to fit on
            target_col: Optional target column name
            
        Returns:
            self: Fitted preprocessor instance for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted parameters.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            pd.DataFrame: Transformed DataFrame
            
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        pass
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data in one step.
        
        This is a convenience method that combines fit() and transform().
        
        Args:
            df: Input DataFrame
            target_col: Optional target column name
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        self.fit(df, target_col)
        return self.transform(df)
    
    def _check_is_fitted(self) -> None:
        """
        Check if the preprocessor has been fitted.
        
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before transforming data. "
                "Call fit() or fit_transform() first."
            )
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the fitted parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of fitted parameters
        """
        return self._fitted_params.copy()
    
    def set_params(self, **params) -> 'BasePreprocessor':
        """
        Set configuration parameters.
        
        Args:
            **params: Key-value pairs of parameters to set
            
        Returns:
            self: Preprocessor instance for method chaining
        """
        self.config.update(params)
        return self
