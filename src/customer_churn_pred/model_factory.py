from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class ModelFactory:
    @staticmethod
    def get_model(model_type, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {}

        if model_type == "random_forest":
            return RandomForestClassifier(**hyperparameters)
        elif model_type == "logistic_regression":
            return LogisticRegression(**hyperparameters)
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(**hyperparameters)
        elif model_type == "svm":
            return SVC(**hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
