from zenml.client import Client
from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Run the training pipeline with the specified data path
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path = "C:\\Users\\jv3250328\\Downloads\\Customer_Satisfaction_MLOps_Project\\data\\olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:C:\Users\jv3250328\AppData\Roaming\zenml\local_stores\4efad8f6-35c2-450f-b80e-a950d69393e1\mlruns"