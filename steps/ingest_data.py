import logging
import pandas as pd
from zenml import step


class IngesData:
    """
    Class to handle data ingestion.
    """
    def __init__(self, data_path: str):
        """
        Args: data_path (str): path to the data file.
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Ingest the data from the data_path.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data_step(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.
    
    Args:
        data_path (str): path to the data file.

    Returns:
        pd.DataFrame: the ingested data as a pandas DataFrame.
    """
    try:
        ingest_data = IngesData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise