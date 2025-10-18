"""
Base Data Loader abstract class.

Provides a common interface for all data loading operations in the project.

Responsibilities:
- Data Engineer: Primary owner - implements concrete loaders for various sources
- Software Engineer: Maintains base class and validates design patterns
- Data Scientist: Uses concrete implementations for data acquisition
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for data loading operations.
    
    This class enforces a consistent interface for loading data from various sources
    (CSV, databases, APIs, cloud storage, etc.) following the Template Method pattern.
    
    Attributes:
        source (Union[str, Path]): The data source location
        config (Dict[str, Any]): Configuration parameters for the loader
    """
    
    def __init__(self, source: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base data loader.
        
        Args:
            source: Path or URI to the data source
            config: Optional configuration dictionary
        """
        self.source = Path(source) if isinstance(source, str) else source
        self.config = config or {}
        logger.info(f"Initialized {self.__class__.__name__} for source: {self.source}")
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data from the source.
        
        This method must be implemented by all concrete data loaders.
        
        Returns:
            pd.DataFrame: The loaded data
            
        Raises:
            FileNotFoundError: If the source doesn't exist
            ValueError: If the data format is invalid
        """
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """
        Validate that the data source exists and is accessible.
        
        Returns:
            bool: True if source is valid, False otherwise
        """
        pass
    
    def load_with_validation(self) -> Optional[pd.DataFrame]:
        """
        Template method that loads data with validation.
        
        This method implements the Template Method pattern, ensuring
        consistent validation before loading.
        
        Returns:
            Optional[pd.DataFrame]: Loaded data or None if validation fails
        """
        if not self.validate_source():
            logger.error(f"Source validation failed for: {self.source}")
            return None
        
        try:
            df = self.load()
            logger.info(f"Successfully loaded {len(df)} rows from {self.source}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {self.source}: {str(e)}")
            return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the data source.
        
        Returns:
            Dict[str, Any]: Metadata including source type, size, etc.
        """
        return {
            'source': str(self.source),
            'loader_type': self.__class__.__name__,
            'config': self.config
        }
