# %% [markdown]
# # Descripcion del proyecto

# %% [markdown]
# La compañía móvil Megaline no está satisfecha al ver que muchos de sus clientes utilizan planes heredados. Quieren desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart o Ultra.
# 
# Tienes acceso a los datos de comportamiento de los suscriptores que ya se han cambiado a los planes nuevos (del proyecto del sprint de Análisis estadístico de datos). Para esta tarea de clasificación debes crear un modelo que escoja el plan correcto.
# 
# Como ya hiciste el paso de procesar los datos, puedes lanzarte directo a crear el modelo.
# Desarrolla un modelo con la mayor exactitud posible. En este proyecto, el umbral de exactitud es 0.75. Usa el dataset para comprobar la exactitud.

# %% [markdown]
# ## Instrucciones del proyecto

# %% [markdown]
# - Abre y examina el archivo de datos. Dirección al archivo:datasets/users_behavior.csv Descarga el dataset
# - Segmenta los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.
# - Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio.
# - Comprueba la calidad del modelo usando el conjunto de prueba.
# - Tarea adicional: haz una prueba de cordura al modelo. Estos datos son más complejos que los que habías usado antes así que no será una tarea fácil. Más adelante lo veremos con más detalle.

# %% [markdown]
# ## Descripcion de datos
# 

# %% [markdown]
# Cada observación en el dataset contiene información del comportamiento mensual sobre un usuario. La información dada es la siguiente:
# 
# - сalls — número de llamadas,
# - minutes — duración total de la llamada en minutos,
# - messages — número de mensajes de texto,
# - mb_used — Tráfico de Internet utilizado en MB,
# - is_ultra — plan para el mes actual (Ultra - 1, Smart - 0).

# %% [markdown]
# ## Evaluación del proyecto

# %% [markdown]
# Hemos definido los criterios de evaluación para el proyecto. Lee esto con atención antes de pasar al ejercicio. 
# Esto es lo que los revisores buscarán cuando evalúen tu proyecto:
# 
# - ¿Cómo leíste los datos después de descargarlos?
# - ¿Segmentaste correctamente los datos en conjuntos de entrenamiento, validación y prueba?
# - ¿Cómo escogiste el tamaño de los conjuntos?
# - ¿Evaluaste correctamente la calidad del modelo?
# - ¿Qué modelos e hiperparámentros usaste?
# - ¿Cuáles fueron tus hallazgos?
# - ¿Probaste los modelos correctamente?
# - ¿Cuál es tu puntuación de exactitud?
# - ¿Te ceñiste a la estructura del proyecto y mantuviste limpio el código?
# 
# Tienes tus hojas informativas y los resúmenes de los capítulos así que ya puedes continuar con el proyecto.
# ¡Buena suerte!

# %% [markdown]
# ### Abrir y examinar el archivo de datos

# %%
# Importar librerias necesarias para el proyecto
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from IPython.print import 
from sklearn.model_selection import GridSearchCV
# %%
# Importar Archivos
df = pd.read_csv('../data/users_behavior.csv')

# Mostrar datos relevantes del dataframe
print(('---------------------------------'))
print(('======= Datos de comportamiento de usuarios ======= '))
print(df.head())
print(df.info())
print(df.describe())

# Verificacion de datos nulos 
nulos = df.isnull().sum()
print(('======= Verificacion de valores nulos ======='))
print(nulos)

# Verificación de valores duplicados
duplicados = df.duplicated().sum()
print(('======= Verificación de valores duplicados ======='))
print(f"Total de valores duplicados: {duplicados}")

print(('---------------------------------'))

comentario = """
======= Comentario sobre la exploracion inicial del archivo =======
Tras Realizar la exploracion inicial de datos se observo que no hay datos duplicados o nulos dentro del conjunto.
Este conjunto de datos recopila informacion sobre el comportamiento de los usuarios, lo cuales engloban las llamadas, minutos utilizados, mensaje enviados y el uso de datos en MB

"""
print((comentario))

# %% [markdown]
# ### Segmentacion de datos en conjuntos de entrenamiento, validacion y prueba

# %%
# Dividir los datos en características (X) y la variable objetivo (y)
features = df.drop('is_ultra', axis=1)
target = df['is_ultra']

# Dividir los datos en conjuntos de entrenamiento (70%), validación (15%), y prueba (15%)
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=12345)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=12345)

# Muestra el tamaño de cada conjunto
print("Tamaño del conjunto de entrenamiento:", len(X_train))
print("Tamaño del conjunto de validación:", len(X_valid))
print("Tamaño del conjunto de prueba:", len(X_test))

# %% [markdown]
# ### Aplicar modelo Random Forest y evaluacion

# %%
# Definir la grilla de hiperparámetros para Random Forest
param_grid_rf = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 4, 7],
    'min_samples_split': [2, 5, 10]
}

# Crear el objeto GridSearchCV
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=12345), 
                              param_grid=param_grid_rf, 
                              cv=5, 
                              scoring='accuracy')

# Ajustar GridSearchCV
grid_search_rf.fit(X_train, y_train)

# Mejores hiperparámetros
print("Mejores hiperparámetros para Random Forest:", grid_search_rf.best_params_)

# Crear una nueva instancia del modelo con los mejores hiperparámetros
best_rf_model = RandomForestClassifier(n_estimators=grid_search_rf.best_params_['n_estimators'],
                                       max_depth=grid_search_rf.best_params_['max_depth'],
                                       min_samples_leaf=grid_search_rf.best_params_['min_samples_leaf'],
                                       min_samples_split=grid_search_rf.best_params_['min_samples_split'],
                                       random_state=12345)

# Entrenar el modelo con el conjunto de entrenamiento
best_rf_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba y evaluar
predictions_test_rf = best_rf_model.predict(X_test)
accuracy_test_rf = accuracy_score(y_test, predictions_test_rf)
print("Exactitud del modelo Random Forest en el conjunto de prueba:", accuracy_test_rf)


# %% [markdown]
# ### Aplicar modelo Logistic Regression y evaluacion

# %%
# Definir la grilla de hiperparámetros para Regresión Logística
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Crear el objeto GridSearchCV
grid_search_lr = GridSearchCV(estimator=LogisticRegression(random_state=12345, max_iter=200), param_grid=param_grid_lr, cv=5, scoring='accuracy')

# Ajustar GridSearchCV
grid_search_lr.fit(X_train, y_train)

# Mejores hiperparámetros
print("Mejores hiperparámetros para Logistic Regression:", grid_search_lr.best_params_)

# Crear una nueva instancia del modelo con los mejores hiperparámetros

best_lr_model = LogisticRegression(C=grid_search_lr.best_params_['C'],
                                       penalty=grid_search_lr.best_params_['penalty'],
                                       solver=grid_search_lr.best_params_['solver'],
                                       random_state=12345)

# Entrenar el modelo con el conjunto de entramiento
best_lr_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba y evaluar
predictions_test_lr = best_lr_model.predict(X_test)
accuracy_test_lr = accuracy_score(y_test, predictions_test_lr)
print("Exactitud del modelo Logistic Regression en el conjunto de prueba:", accuracy_test_lr)



# %% [markdown]
# ### Prueba de cordura

# %%
# Analiza la importancia de las características
rf_feature_importances = pd.DataFrame(best_rf_model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)


# Obtener los coeficientes del modelo
lr_coefficients = best_lr_model.coef_[0]

# Crear un DataFrame para visualizar los coeficientes (importancia) de cada característica
lr_feature_importance = pd.DataFrame(lr_coefficients, 
                                     index=X_train.columns, 
                                     columns=['Coefficient']).sort_values('Coefficient', ascending=False)

print(('---------------------------------'))
print(('======= Random forest: '))
print("Importancia de características en Random Forest:\n", rf_feature_importances)
print(('---------------------------------'))
print(('======= Logistic Regression: '))
print("Importancia de las características en la Regresión Logística:\n", lr_feature_importance)




