# src/main.py
from preprocessing.preprocess import load_and_clean_data, split_data
from models.random_forest import train_random_forest
from models.logistic_regression import train_logistic_regression
from evaluation.model_evaluation import evaluate_model
from utils.generate_report import save_report_as_html

# Cargar y limpiar los datos
df = load_and_clean_data('data/users_behavior.csv')

# Dividir los datos
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

# Entrenar modelos
rf_model = train_random_forest(X_train, y_train)
lr_model = train_logistic_regression(X_train, y_train)

# Evaluar modelos
accuracy_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")

accuracy_lr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

save_report_as_html(accuracy_rf, accuracy_lr)