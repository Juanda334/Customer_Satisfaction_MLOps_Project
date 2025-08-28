import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model
        Args: 
            X_train: training features
            y_train: training labels
        Returns: 
            None
        """
        pass
    
class LinearRegressionModel(Model):
    """
    Linear Regression Model
    Args:
        Model (ABC): Abstract class for all models
    Returns: 
        None
    """
        
    def train(self, X_train, y_train, **kwargs):
        """
        Train the Linear Regression model
        Args: 
            X_train: training features
            y_train: training labels
        Returns: 
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error while training Linear Regression model: {e}")
            raise