import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from src.evaluation import R2, RMSE

@step
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
        df (pd.DataFrame): The data to evaluate the model on.
    """
    try:
        prediction = model.predict(X_test)
        
        r2_score = R2().calculate_scores(y_true=y_test, y_pred=prediction)
        
        rmse = RMSE().calculate_scores(y_true=y_test, y_pred=prediction)
        
        return r2_score, rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e