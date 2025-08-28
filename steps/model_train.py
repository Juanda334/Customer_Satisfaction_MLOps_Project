import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel

@step(enable_cache = False, experiment_tracker="mlflow_tracker")  # Use MLflow experiment tracker
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    ) -> RegressorMixin:
    """
    Train a machine learning model.
    Args: 
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features  
        y_train (pd.Series): Training targets
        y_test (pd.Series): Test targets
    Returns:
        RegressorMixin: Trained model
    """
    try:
        # Enable MLflow autologging to track the model automatically
        mlflow.sklearn.autolog()
        
        # Train the model
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        
        # Log additional parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", list(X_train.columns))
        
        print("Linear Regression model trained successfully")
        
        # Return the trained model as a ZenML artifact
        return trained_model
        
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise
