import pytest
from sklearn.datasets import make_classification
from customer_churn_pred.model import ChurnModel

@pytest.mark.parametrize(
    "model_type, hyperparameters",
    [
        ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
        ("logistic_regression", {"solver": "lbfgs", "max_iter": 200}),
        ("gradient_boosting", {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 5}),
        ("svm", {"kernel": "rbf", "C": 1.0}),
    ],
)
def test_model_training(model_type, hyperparameters):
    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # Initialize and train the model
    model = ChurnModel(model_type=model_type, hyperparameters=hyperparameters)
    model.train(X, y)
    y_pred = model.predict(X)

    # Ensure predictions have the correct length
    assert len(y_pred) == len(y), f"Prediction length mismatch for model: {model_type}"
