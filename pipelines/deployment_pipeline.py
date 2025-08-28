import json
import numpy as np
import pandas as pd
from pydantic import BaseModel
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.cleaning_data import clean_data_step
from steps.evaluation import model_evaluation
from steps.ingest_data import ingest_data_step
from steps.model_train import train_model
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    """
    Deployment trigger config
    """
    min_accuracy: float = 0

@step(enable_cache = False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """
    Implements a simple model deployment trigger that looks at the input model accuracy and decide if it's good enough to deploy.
    """
    return accuracy >= config.min_accuracy

@step(enable_cache = False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = False,  # Changed to False for Windows compatibility
    model_name: str = "model"
) -> MLFlowDeploymentService:
    """
    Get the predictions service started by the deployment pipeline
    
    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction,
        pipeline_step_name: the name of the step that deployed the MLflow prediction,
        running: when this is flag is set, the step only returns a running service,
        model_name: the name of the model that is deployed
    """
    # Get the MLflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    
    # fetch existing services with same pipeline name, step, name and model name
    # First try to find running services
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name = model_name,
        running = True
    )
    
    # If no running services found, try to find any services (including stopped ones)
    if not existing_services:
        existing_services = mlflow_model_deployer_component.find_model_server(
            pipeline_name = pipeline_name,
            pipeline_step_name = pipeline_step_name,
            model_name = model_name,
            running = False
        )
    
    if not existing_services:
        raise RuntimeError(
            f"No ML deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}. "
            f"Please run the deployment pipeline first."
        )
    
    service = existing_services[0]
    
    # For Windows compatibility, don't require the service to be running as a daemon
    # Instead, we'll start it when needed in the predictor step
    return service

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray
) -> np.ndarray:
    
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.0,  # Set to 0.0 to allow deployment with any accuracy for testing
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Initialize deployment trigger config
    deployment_config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    
    # Pipeline steps
    df = ingest_data_step(data_path)
    X_train, X_test, y_train, y_test = clean_data_step(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = model_evaluation(model, X_test, y_test)
    
    # Use r2_score for deployment decision (assuming higher is better)
    # If you want to use RMSE instead, you'd need to invert the logic since lower RMSE is better
    deployment_decision = deployment_trigger(
        accuracy=r2_score,
        config=deployment_config,
    )
    
    if deployment_decision:
        mlflow_model_deployer_step(
            model=model,
            workers=workers,
            timeout=timeout,
            mlserver=False,  # Use MLflow's built-in server instead of MLServer
        )
        
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=True,
        model_name="model",
    )
    predictor(service=model_deployment_service, data=batch_data)
