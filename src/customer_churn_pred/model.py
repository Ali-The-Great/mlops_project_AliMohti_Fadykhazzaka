from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from customer_churn_pred.model_factory import ModelFactory
from customer_churn_pred.evaluation_strategy import ClassificationReportStrategy, AccuracyStrategy, F1ScoreStrategy


class ChurnModel:
       def __init__(self, model_type="random_forest", hyperparameters=None, evaluation_strategy=None):
        self.model = ModelFactory.get_model(model_type, hyperparameters)
        self.evaluation_strategy = evaluation_strategy or ClassificationReportStrategy()        
       def __init__(self, model_type="random_forest", hyperparameters=None):
        self.model = ModelFactory.get_model(model_type, hyperparameters)
       def __init__(self, model_type="random_forest", hyperparameters=None ):
        
        # Set default hyperparameters if none are provided
        if hyperparameters is None:
            hyperparameters = {}

        # Initialize the model based on the type and hyperparameters
        if model_type == "random_forest":
            self.model = RandomForestClassifier(**hyperparameters)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(**hyperparameters)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(**hyperparameters)
        elif model_type == "svm":
            self.model = SVC(**hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

       def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

       def predict(self, X_test):
        return self.model.predict(X_test)

    # def evaluate(self, y_test, y_pred):
    #     report = classification_report(y_test, y_pred, output_dict=True)
    #     return report
    #    def evaluate(self, y_test, y_pred):
    #     from sklearn.metrics import classification_report
    #     report = classification_report(y_test, y_pred, output_dict=True)
    #     return report
   
       def evaluate(self, y_test, y_pred):
        return self.evaluation_strategy.evaluate(y_test, y_pred)