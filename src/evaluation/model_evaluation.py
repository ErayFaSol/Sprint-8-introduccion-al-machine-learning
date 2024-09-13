# src/evaluation/model_evaluation.py
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Exactitud del modelo {model_name}: {accuracy}")
    return accuracy
