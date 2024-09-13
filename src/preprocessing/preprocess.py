# src/preprocessing/preprocess.py
import pandas as pd

def load_and_clean_data(filepath):
    # Cargar los datos
    df = pd.read_csv(filepath)

    # Verificación de nulos y duplicados
    if df.isnull().sum().any() or df.duplicated().sum() > 0:
        print("Datos nulos o duplicados encontrados")

    return df

def split_data(df):
    # Separar las características y la variable objetivo
    features = df.drop('is_ultra', axis=1)
    target = df['is_ultra']

    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=12345)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=12345)

    return X_train, X_valid, X_test, y_train, y_valid, y_test
