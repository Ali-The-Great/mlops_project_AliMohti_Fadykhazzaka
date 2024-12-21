from customer_churn_pred.model import ChurnModel

def make_inference(X_input):
    model = ChurnModel()
    model.load_model('path_to_model')
    predictions = model.predict(X_input)
    return predictions