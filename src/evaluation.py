import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):
    """
    Abstract base class for defining the model evaluation.
    """
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model on the test data.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
        
        Returns:
            float: The evaluation metric value.
        """
        pass

class MSE(Evaluation):
    """
    Mean Squared Error evaluation class.
    """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Implementation of the abstract evaluate method.
        """
        return self.calculate_scores(y_true, y_pred)
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model using Mean Squared Error.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        
        Returns:
            float: The MSE metric value.
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 Score
    """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Implementation of the abstract evaluate method.
        """
        return self.calculate_scores(y_true, y_pred)
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Implementation of the abstract evaluate method.
        """
        return self.calculate_scores(y_true, y_pred)
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e