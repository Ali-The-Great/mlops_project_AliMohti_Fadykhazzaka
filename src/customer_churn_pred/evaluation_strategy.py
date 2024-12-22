from sklearn.metrics import classification_report, accuracy_score, f1_score

class EvaluationStrategy:
    def evaluate(self, y_test, y_pred):
        raise NotImplementedError("Subclasses must implement `evaluate`")

class ClassificationReportStrategy(EvaluationStrategy):
    def evaluate(self, y_test, y_pred):
        return classification_report(y_test, y_pred, output_dict=True)

class AccuracyStrategy(EvaluationStrategy):
    def evaluate(self, y_test, y_pred):
        return {"accuracy": accuracy_score(y_test, y_pred)}

class F1ScoreStrategy(EvaluationStrategy):
    def evaluate(self, y_test, y_pred):
        return {"f1_score": f1_score(y_test, y_pred, average="weighted")}
