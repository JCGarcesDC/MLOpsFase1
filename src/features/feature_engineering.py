"""
Feature Engineering Module - Refactored with OOP.

This module provides feature engineering transformations
following SOLID principles.

Role Assignments:
- Data Scientist: Primary owner - designs new features
- ML Engineer: Integrates features into training pipelines
- Data Engineer: Ensures feature computation efficiency
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import logging

from src.base.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class BMICalculator(BasePreprocessor):
    """
    Calculate Body Mass Index (BMI) feature.
    
    Responsibilities:
    - Data Scientist: Validates BMI calculation logic
    - ML Engineer: Integrates into feature pipeline
    
    Formula: BMI = weight (kg) / height (m)²
    
    Example:
        >>> bmi_calc = BMICalculator(weight_col='weight', height_col='height')
        >>> df_with_bmi = bmi_calc.fit_transform(df)
    """
    
    def __init__(
        self,
        weight_col: str = 'weight',
        height_col: str = 'height',
        output_col: str = 'imc',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BMI calculator.
        
        Args:
            weight_col: Name of weight column (in kg)
            height_col: Name of height column (in meters)
            output_col: Name for the output BMI column
            config: Additional configuration
        """
        super().__init__(config)
        self.weight_col = weight_col
        self.height_col = height_col
        self.output_col = output_col
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'BMICalculator':
        """
        Fit the calculator (validates required columns exist).
        
        Args:
            df: Input DataFrame
            target_col: Not used, present for interface compatibility
            
        Returns:
            self: Fitted calculator
        """
        # Validate required columns exist
        missing_cols = []
        if self.weight_col not in df.columns:
            missing_cols.append(self.weight_col)
        if self.height_col not in df.columns:
            missing_cols.append(self.height_col)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self._fitted_params = {
            'weight_col': self.weight_col,
            'height_col': self.height_col,
            'output_col': self.output_col
        }
        
        self.is_fitted = True
        logger.info("BMICalculator fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add BMI column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with BMI column added
        """
        self._check_is_fitted()
        
        df_out = df.copy()
        df_out[self.output_col] = df_out[self.weight_col] / (df_out[self.height_col] ** 2)
        
        logger.info(f"BMI column '{self.output_col}' created")
        return df_out


class FeatureEngineeringPipeline:
    """
    Orchestrates multiple feature engineering steps.
    
    Responsibilities:
    - ML Engineer: Primary owner - manages feature pipeline
    - Data Scientist: Adds new feature transformers
    
    This class uses the Chain of Responsibility pattern to
    apply multiple transformations in sequence.
    
    Example:
        >>> pipeline = FeatureEngineeringPipeline([
        ...     BMICalculator(),
        ...     AgeGroupEncoder(),
        ...     InteractionFeatures()
        ... ])
        >>> df_engineered = pipeline.fit_transform(df)
    """
    
    def __init__(self, transformers: List[BasePreprocessor]):
        """
        Initialize feature engineering pipeline.
        
        Args:
            transformers: List of feature transformers to apply in order
        """
        self.transformers = transformers
        self.is_fitted = False
        logger.info(f"FeatureEngineeringPipeline initialized with {len(transformers)} transformers")
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'FeatureEngineeringPipeline':
        """
        Fit all transformers in the pipeline.
        
        Args:
            df: Training DataFrame
            target_col: Optional target column
            
        Returns:
            self: Fitted pipeline
        """
        logger.info("Fitting FeatureEngineeringPipeline...")
        
        df_current = df.copy()
        for i, transformer in enumerate(self.transformers):
            logger.info(f"Fitting transformer {i+1}/{len(self.transformers)}: {transformer.__class__.__name__}")
            transformer.fit(df_current, target_col)
            df_current = transformer.transform(df_current)
        
        self.is_fitted = True
        logger.info("FeatureEngineeringPipeline fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformers to the data.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() or fit_transform() first.")
        
        logger.info("Transforming with FeatureEngineeringPipeline...")
        
        df_out = df.copy()
        for i, transformer in enumerate(self.transformers):
            logger.debug(f"Applying transformer {i+1}: {transformer.__class__.__name__}")
            df_out = transformer.transform(df_out)
        
        logger.info("FeatureEngineeringPipeline transformation complete")
        return df_out
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            target_col: Optional target column
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        self.fit(df, target_col)
        return self.transform(df)
