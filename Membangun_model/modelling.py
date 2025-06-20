import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5001/")

# Create a new MLflow Experiment
mlflow.set_experiment("MSML_Hafiz-Caniago Train Predictive Model Boston Housing")

data = pd.read_csv("boston_housing_preprocessing.csv")

# Ensure integer columns are cast to float64 to handle missing values
data = data.astype({col: "float64" for col in data.select_dtypes("int").columns})

X = data.drop(columns=["medv", "CHAS_label", "crime_category", "price_category"])
y = data["medv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Set random seed for reproducibility
input_example = X_train[0:5]

with mlflow.start_run():
    # Log parameters
    n_estimators = 505
    max_depth = 37
    random_state = 42
    mlflow.autolog()
    
    # Train model   
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Log the fitted model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="model_rf_hc",
        input_example=input_example
    )
