import mlflow
from sklearn.datasets import load_breast_cancer

MODEL_URI="models://forge-ml-classifier@champion"

def main():
    model = mlflow.pyfunc.load_model(MODEL_URI)
    dataset = load_breast_cancer()
    sample = dataset.data[:1]
    prediction = model.predict(sample)
    print(f"Model URI: {MODEL_URI}")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
