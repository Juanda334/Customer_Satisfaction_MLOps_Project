import logging
import pandas as pd
import mlflow
from zenml import step
from typing import Tuple
from zenml.client import Client
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from src.evaluation import R2, RMSE

@step(enable_cache = False)
def model_evaluation(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
    ) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"]
    ]:
    """
    Evaluate the trained model.
    Args:
        model (RegressorMixin): The trained model to evaluate
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
    Returns:
        Tuple[float, float]: R2 score and RMSE values
    """
    try:
        # Get predictions
        prediction = model.predict(X_test)
        
        # Calculate metrics
        r2_score = R2().calculate_scores(y_true=y_test, y_pred=prediction)
        rmse = RMSE().calculate_scores(y_true=y_test, y_pred=prediction)
        
        # Log metrics to experiment tracker if available
        client = Client()
        experiment_tracker = client.active_stack.experiment_tracker
        
        if experiment_tracker is not None:
            mlflow.set_experiment("continuous_deployment_pipeline")
            with mlflow.start_run(nested=True) as run:
                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_param("test_samples", len(X_test))

        return r2_score, rmse
        
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e