from src.train import load_data, train_model

def test_model_can_train():
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    assert model is not None
