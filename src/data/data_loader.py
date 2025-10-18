"""
Data Loading Module - Refactored with OOP.

This module provides concrete implementations of data loaders
for different data sources (CSV, DVC, databases, APIs).

Role Assignments:
- Data Engineer: Primary owner - maintains and extends data loaders
- Software Engineer: Reviews architecture and design patterns
- Data Scientist: Uses loaders for data acquisition
- ML Engineer: Integrates loaders into pipelines
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import logging

from src.base.base_data_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class CSVDataLoader(BaseDataLoader):
    """
    Concrete data loader for CSV files.
    
    Responsibilities:
    - Data Engineer: Maintains CSV reading logic and encoding handling
    - Software Engineer: Ensures proper error handling
    
    Example:
        >>> loader = CSVDataLoader('data/raw/obesity_data.csv')
        >>> df = loader.load_with_validation()
    """
    
    def __init__(
        self,
        source: str,
        encoding: str = 'utf-8',
        sep: str = ',',
        **kwargs
    ):
        """
        Initialize CSV data loader.
        
        Args:
            source: Path to CSV file
            encoding: File encoding
            sep: Column separator
            **kwargs: Additional pandas read_csv parameters
        """
        config = {'encoding': encoding, 'sep': sep, **kwargs}
        super().__init__(source, config)
    
    def validate_source(self) -> bool:
        """
        Validate that the CSV file exists.
        
        Returns:
            bool: True if file exists
        """
        if not self.source.exists():
            logger.error(f"CSV file not found: {self.source}")
            return False
        
        if not self.source.suffix.lower() in ['.csv', '.txt']:
            logger.warning(f"Unexpected file extension: {self.source.suffix}")
        
        return True
    
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If CSV parsing fails
        """
        try:
            df = pd.read_csv(
                self.source,
                encoding=self.config.get('encoding', 'utf-8'),
                sep=self.config.get('sep', ','),
                **{k: v for k, v in self.config.items() if k not in ['encoding', 'sep']}
            )
            
            logger.info(
                f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns from {self.source.name}"
            )
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {self.source}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV {self.source}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {self.source}: {str(e)}")
            raise


class DataFrameAnalyzer:
    """
    Utility class for DataFrame analysis and exploration.
    
    Responsibilities:
    - Data Scientist: Primary user - performs exploratory data analysis
    - Data Engineer: Maintains data quality checks
    
    This class encapsulates common EDA operations in a reusable,
    testable format.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with a DataFrame.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        logger.info(f"Initialized analyzer for DataFrame: {df.shape}")
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the DataFrame.
        
        Returns:
            Dict containing shape, columns, dtypes, memory usage
        """
        return {
            'shape': self.df.shape,
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def get_missing_values_summary(self) -> pd.DataFrame:
        """
        Get summary of missing values.
        
        Returns:
            pd.DataFrame with columns: column, missing_count, missing_percentage
        """
        missing = pd.DataFrame({
            'missing_count': self.df.isnull().sum(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        missing = missing[missing['missing_count'] > 0].sort_values(
            'missing_count', ascending=False
        )
        return missing
    
    def get_duplicate_summary(self) -> Dict[str, Any]:
        """
        Get summary of duplicate rows.
        
        Returns:
            Dict containing duplicate count and percentage
        """
        n_duplicates = self.df.duplicated().sum()
        return {
            'n_duplicates': int(n_duplicates),
            'duplicate_percentage': round(n_duplicates / len(self.df) * 100, 2)
        }
    
    def get_numeric_columns(self) -> List[str]:
        """
        Get list of numeric column names.
        
        Returns:
            List of numeric column names
        """
        return self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def get_categorical_columns(self) -> List[str]:
        """
        Get list of categorical column names.
        
        Returns:
            List of categorical column names
        """
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.
        
        Returns:
            pd.DataFrame with descriptive statistics
        """
        numeric_cols = self.get_numeric_columns()
        if not numeric_cols:
            logger.warning("No numeric columns found")
            return pd.DataFrame()
        
        return self.df[numeric_cols].describe().T
    
    def get_categorical_value_counts(self, top_n: int = 10) -> Dict[str, pd.Series]:
        """
        Get value counts for categorical columns.
        
        Args:
            top_n: Number of top values to return per column
            
        Returns:
            Dict mapping column names to their value counts
        """
        cat_cols = self.get_categorical_columns()
        return {
            col: self.df[col].value_counts().head(top_n)
            for col in cat_cols
        }
    
    def generate_eda_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive EDA report.
        
        Returns:
            Dict containing all analysis results
        """
        report = {
            'basic_info': self.get_basic_info(),
            'missing_values': self.get_missing_values_summary().to_dict(),
            'duplicates': self.get_duplicate_summary(),
            'numeric_columns': self.get_numeric_columns(),
            'categorical_columns': self.get_categorical_columns(),
            'summary_statistics': self.get_summary_statistics().to_dict()
        }
        
        logger.info("Generated comprehensive EDA report")
        return report


class VariableManager:
    """
    Manages variable categorization and configuration.
    
    Responsibilities:
    - Data Scientist: Defines variable groupings
    - Data Engineer: Maintains variable metadata
    
    This class centralizes variable definitions to avoid hardcoding
    throughout the codebase (DRY principle).
    """
    
    # Default variable definitions for obesity dataset
    DEFAULT_NUMERIC_VARS = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 
        'CH2O', 'FAF', 'TUE'
    ]
    
    DEFAULT_CATEGORICAL_VARS = [
        'Gender', 'family_history_with_overweight', 'FAVC', 
        'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]
    
    DEFAULT_TARGET_VAR = 'NObeyesdad'
    
    def __init__(
        self,
        numeric_vars: Optional[List[str]] = None,
        categorical_vars: Optional[List[str]] = None,
        target_var: Optional[str] = None,
        to_lower: bool = False
    ):
        """
        Initialize variable manager.
        
        Args:
            numeric_vars: Custom list of numeric variable names
            categorical_vars: Custom list of categorical variable names
            target_var: Custom target variable name
            to_lower: Whether to convert all names to lowercase
        """
        self.numeric_vars = numeric_vars or self.DEFAULT_NUMERIC_VARS.copy()
        self.categorical_vars = categorical_vars or self.DEFAULT_CATEGORICAL_VARS.copy()
        self.target_var = target_var or self.DEFAULT_TARGET_VAR
        
        if to_lower:
            self.numeric_vars = [v.lower() for v in self.numeric_vars]
            self.categorical_vars = [v.lower() for v in self.categorical_vars]
            self.target_var = self.target_var.lower()
        
        logger.info(
            f"VariableManager initialized: "
            f"{len(self.numeric_vars)} numeric, "
            f"{len(self.categorical_vars)} categorical"
        )
    
    def get_all_feature_vars(self) -> List[str]:
        """Get all feature variables (numeric + categorical)."""
        return self.numeric_vars + self.categorical_vars
    
    def get_all_vars(self) -> List[str]:
        """Get all variables including target."""
        return self.get_all_feature_vars() + [self.target_var]
    
    def exclude_variables(self, vars_to_exclude: List[str]) -> 'VariableManager':
        """
        Create a new VariableManager with specified variables excluded.
        
        Args:
            vars_to_exclude: List of variable names to exclude
            
        Returns:
            New VariableManager instance
        """
        new_numeric = [v for v in self.numeric_vars if v not in vars_to_exclude]
        new_categorical = [v for v in self.categorical_vars if v not in vars_to_exclude]
        
        return VariableManager(
            numeric_vars=new_numeric,
            categorical_vars=new_categorical,
            target_var=self.target_var
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dict with variable groupings
        """
        return {
            'numeric_vars': self.numeric_vars,
            'categorical_vars': self.categorical_vars,
            'target_var': self.target_var,
            'n_numeric': len(self.numeric_vars),
            'n_categorical': len(self.categorical_vars)
        }
