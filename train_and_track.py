import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Training
def train_model(X_train, y_train, X_test, y_test, params):
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return lr, accuracy

# Tracking
def track_model(params, accuracy, X_train, lr):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    mlflow.set_experiment("MLflow Quickstart")

    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        signature = infer_signature(X_train, lr.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )

if __name__ == '__main__':
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 42, # change one hyperparameter. Old one: 8888,
    }

    model, accuracy = train_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, params=params)
    track_model(params=params, accuracy=accuracy, X_train=X_train, lr=model)