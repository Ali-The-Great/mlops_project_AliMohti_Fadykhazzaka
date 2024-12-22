import yaml
import os
from sklearn.model_selection import train_test_split
from customer_churn_pred.data_preprocessing import load_data, preprocess_data
from customer_churn_pred.model import ChurnModel
from customer_churn_pred.logger import get_logger
from customer_churn_pred.experiment_tracker import log_experiment

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
logger = get_logger()

def train_model(config_path='src/customer_churn_pred/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load and preprocess data
    df = load_data(config['data_url'])
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=config['random_state'])

    # Extract model type and hyperparameters
    model_type = config['model']['type']
    hyperparameters = config['model'].get('hyperparameters', {})
    logger.info(f"Training model: {model_type} with hyperparameters: {hyperparameters}")

    # Initialize and train model
    model = ChurnModel(model_type=model_type, hyperparameters=hyperparameters)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    report = model.evaluate(y_test, y_pred)
    logger.info(f"Evaluation Report: {report}")

    # Log the experiment
    log_experiment(model, X_train, y_train, report)

if __name__ == "__main__":
    train_model()
