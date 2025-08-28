from zenml import pipeline
from steps.ingest_data import ingest_data_step
from steps.cleaning_data import clean_data_step
from steps.model_train import train_model
from steps.evaluation import model_evaluation

@pipeline(enable_cache = False)
def training_pipeline(data_path: str):
    """
    A training pipeline that includes data ingestion, preprocessing, model training, and evaluation steps.
    """
    df = ingest_data_step(data_path)
    X_train, X_test, y_train, y_test = clean_data_step(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = model_evaluation(model, X_test, y_test)
