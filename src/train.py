import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

from src.utils.mlflow_utils import setup_mlflow, log_model_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config")
def train(config: DictConfig):
    """Main training pipeline."""
    # Setup MLflow
    experiment_id = setup_mlflow(config.mlflow.experiment_name)
    mlflow.set_experiment_id(experiment_id)
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "model_type": "xgboost",
            "random_state": config.model.random_state,
            "test_size": config.model.test_size
        })
        
        # Your training code here
        logger.info("Starting training pipeline...")
        
        # Example: Load data
        # df = pd.read_csv(Path(config.data.processed) / "training_data.csv")
        
        # Example: Train model
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, 
        #     test_size=config.model.test_size,
        #     random_state=config.model.random_state
        # )
        
        # Log metrics
        # mlflow.log_metrics({
        #     "accuracy": accuracy,
        #     "precision": precision,
        #     "recall": recall,
        #     "f1": f1
        # })
        
        # Log model
        # mlflow.sklearn.log_model(model, "model")
        
        # Register model
        log_model_info(run.info.run_id, config.mlflow.model_name)
        
        logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    train()