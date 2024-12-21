import mlflow
import mlflow.sklearn

def log_experiment(model, X_train, y_train, report):
    with mlflow.start_run():
        mlflow.log_params({
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        })
        mlflow.log_metrics({'accuracy': report['accuracy']})
        mlflow.sklearn.log_model(model, "model")