import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    ) -> RegressorMixin:
    """
    Train a machine learning model.
    Args: 
        df (pd.DataFrame): The training data.
    """
    try:
        model = LinearRegressionModel()
        model_trained = model.train(X_train, y_train)
        return model_trained

    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise