from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Configuration parameters for the pipeline.
    """
    model_name: str = "LinearRegression"