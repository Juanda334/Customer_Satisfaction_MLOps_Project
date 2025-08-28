import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.data_cleaning import DaraPreprocessStrategy, DataDivideStrategy, DataCleaning

@step
def clean_data_step(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Clean the data by preprocessing and dividing into features and target.
    
    Args:
        df (pd.DataFrame): the input data as a pandas DataFrame.
        
    Returns:
        X_train (pd.DataFrame): training features.
        X_test (pd.DataFrame): testing features.
        y_train (pd.Series): training target.
        y_test (pd.Series): testing target.
    """
    try:
        process_strategy = DaraPreprocessStrategy()
        data_cleaning = DataCleaning(data = df, strategy = process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(data = processed_data, strategy = divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning and division completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data cleaning step: {e}")
        raise