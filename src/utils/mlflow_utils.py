from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(experiment_name: str = "obesity_prediction"):
    """Setup MLflow tracking and returns experiment ID."""
    # Set tracking URI to local 'mlruns' directory
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=str(Path("mlruns") / experiment_name)
        )
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    return experiment_id

def log_model_info(run_id: str, model_name: str):
    """Log model info to MLflow registry."""
    client = MlflowClient()
    
    # Register model if not exists
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException:
        pass
    
    # Create model version
    client.create_model_version(
        name=model_name,
        source=f"mlruns/0/{run_id}/artifacts/model",
        run_id=run_id,
    )