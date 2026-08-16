from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

N_ESTIMATORS = 500
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    dataset = load_breast_cancer()
    return train_test_split(dataset.data, dataset.target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=dataset.target)


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


def save_model(model):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def main():
    # Experiment that the subsequent runs should belong to.
    mlflow.set_experiment("forge-ml-training")
    X_train, X_test, y_train, y_test = load_data()
    # Start the MLFlow run, active only under with block
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        save_model(model)
        # Hyperparameters
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        # This is not just logging the metadata about the model, its
        # logging the trained model itself, including the fitted parameters
        # learned when calling .fit() function in train_model()
        mlflow.sklearn.log_model(sk_model=model, name="model")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model saved locally to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
