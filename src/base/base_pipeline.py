"""
Base Pipeline abstract class.

Provides a common interface for end-to-end ML pipelines.

Responsibilities:
- ML Engineer: Primary owner - orchestrates the entire ML workflow
- Data Engineer: Implements data pipeline steps
- Software Engineer: Maintains pipeline architecture and error handling
- Data Scientist: Configures and runs pipelines for experiments
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """
    Abstract base class for end-to-end ML pipelines.
    
    This class orchestrates the complete workflow from data loading
    through model training and evaluation, following the Chain of
    Responsibility pattern.
    
    Attributes:
        pipeline_name (str): Name of the pipeline
        config (Dict[str, Any]): Pipeline configuration
        steps (List[str]): List of pipeline step names
    """
    
    def __init__(self, pipeline_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base pipeline.
        
        Args:
            pipeline_name: Name identifier for the pipeline
            config: Optional configuration dictionary
        """
        self.pipeline_name = pipeline_name
        self.config = config or {}
        self.steps = []
        self._results = {}
        self._artifacts = {}
        logger.info(f"Initialized {self.__class__.__name__}: {pipeline_name}")
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load data for the pipeline.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        pass
    
    @abstractmethod
    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        pass
    
    @abstractmethod
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Any: Trained model
        """
        pass
    
    @abstractmethod
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        pass
    
    def run(self, save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        This is the main orchestration method that calls all pipeline steps
        in the correct order and handles errors gracefully.
        
        Args:
            save_artifacts: Whether to save pipeline artifacts
            
        Returns:
            Dict[str, Any]: Pipeline results including metrics and artifacts
        """
        logger.info(f"Starting pipeline: {self.pipeline_name}")
        
        try:
            # Step 1: Load data
            logger.info("Step 1/5: Loading data...")
            df = self.load_data()
            self._results['data_shape'] = df.shape
            
            # Step 2: Preprocess data
            logger.info("Step 2/5: Preprocessing data...")
            df_processed = self.preprocess_data(df)
            self._results['processed_shape'] = df_processed.shape
            
            # Step 3: Split data
            logger.info("Step 3/5: Splitting data...")
            X_train, X_test, y_train, y_test = self.split_data(df_processed)
            self._results['train_size'] = len(X_train)
            self._results['test_size'] = len(X_test)
            
            # Step 4: Train model
            logger.info("Step 4/5: Training model...")
            model = self.train_model(X_train, y_train)
            self._artifacts['model'] = model
            
            # Step 5: Evaluate model
            logger.info("Step 5/5: Evaluating model...")
            metrics = self.evaluate_model(model, X_test, y_test)
            self._results['metrics'] = metrics
            
            # Save artifacts if requested
            if save_artifacts:
                self.save_artifacts()
            
            logger.info(f"Pipeline {self.pipeline_name} completed successfully")
            return self._results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_artifacts(self, output_dir: Optional[Path] = None) -> None:
        """
        Save pipeline artifacts (models, results, etc.).
        
        Args:
            output_dir: Optional custom output directory
        """
        if output_dir is None:
            output_dir = Path(self.config.get('output_dir', 'artifacts'))
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving artifacts to {output_dir}")
        # Subclasses should implement specific artifact saving logic
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get pipeline execution results.
        
        Returns:
            Dict[str, Any]: Results dictionary
        """
        return self._results.copy()
    
    def get_artifacts(self) -> Dict[str, Any]:
        """
        Get pipeline artifacts.
        
        Returns:
            Dict[str, Any]: Artifacts dictionary
        """
        return self._artifacts.copy()
