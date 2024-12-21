from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class ChurnModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        return report