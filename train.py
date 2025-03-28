
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import numpy as np
import mlflow
import mlflow.sklearn

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default='/opt/ml/output/data')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--test', type=str, default='/opt/ml/input/data/test')
    
    args = parser.parse_args()
    
    try:
        # Increase timeout for MLflow HTTP requests
        os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '300'  # Increase timeout to 300 seconds
        mlflow.set_tracking_uri("http://52.207.216.244:5000")
        
        # Start an MLflow run
        mlflow.start_run()

        # Load the train and test datasets from S3
        train_data = pd.read_csv(os.path.join(args.train, "train_data.csv"))
        test_data = pd.read_csv(os.path.join(args.test, "test_data.csv"))

        # Handle NaN and infinite values
        train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)

        # Split features and target
        X_train = train_data.drop(columns='Class')
        y_train = train_data['Class']
        X_test = test_data.drop(columns='Class')
        y_test = test_data['Class']

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1}")

        # Log the model with MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("f1_score", f1)

        # Save the model regardless of the F1 score
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

        # End the MLflow run
        mlflow.end_run()
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

