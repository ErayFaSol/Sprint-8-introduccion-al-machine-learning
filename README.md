# Introducción al Machine Learning

Este proyecto se enfoca en desarrollar un modelo de clasificación para recomendar planes de telefonía móvil basados en el comportamiento de los clientes. Utiliza técnicas de machine learning supervisado para predecir el plan más adecuado para cada cliente en función de su perfil de uso.

## Tecnologías utilizadas
- Python
- pandas
- scikit-learn

## Objetivo
Utilizar datos históricos del comportamiento de los clientes para recomendar el plan de telefonía móvil más adecuado, mejorando la experiencia del cliente y aumentando la retención.

## Contexto
La empresa de telecomunicaciones desea optimizar las recomendaciones de planes de telefonía móvil para sus clientes actuales y potenciales. En lugar de ofrecer todos los planes de manera genérica, se busca personalizar la oferta en base a los hábitos de consumo de cada cliente, como el uso de datos, llamadas y mensajes de texto.

## Descripción del Proyecto
El análisis se basa en un conjunto de datos que contiene información sobre el uso de diferentes servicios por parte de los clientes de una empresa de telecomunicaciones. La variable objetivo es el plan recomendado (Ultra o Smart), y las variables independientes son los patrones de uso de datos, llamadas y mensajes de texto.

Se sigue el siguiente flujo de trabajo:

1. **Carga y exploración de datos**: Se analiza el conjunto de datos para identificar patrones y relaciones entre las variables.
   
2. **Preprocesamiento de datos**: Limpieza y transformación de los datos para que puedan ser utilizados por los modelos.
   
3. **División de los datos**: Separación en conjuntos de entrenamiento, validación y prueba.
   
4. **Entrenamiento del modelo**: Se entrenaron dos modelos de clasificación, Random Forest y Regresión Logística, ajustando hiperparámetros mediante GridSearchCV.
   
5. **Evaluación del modelo**: Se utilizan métricas como la precisión para evaluar el rendimiento.
   
6. Los resultados del modelo se guardan en un archivo HTML llamado reporte_final.html.

## Entrenamiento del modelo

se utilizaron dos modelos de clasificacion: 

- *Random Forest:* Se probaron diferentes hiperparámetros, como la profundidad máxima y el número de estimadores, obteniendo una precisión del 82%.
- *Regresión Logística:* También se ajustaron hiperparámetros como el parámetro de regularización, obteniendo una precisión del 78%.

## Conclusiones
El modelo de Random Forest logró recomendar correctamente el plan en el 82% de los casos. Este resultado puede ser utilizado para personalizar las ofertas de la empresa de telecomunicaciones y mejorar la retención de clientes.

### Futuras mejoras
- Incluir más características relacionadas con el perfil de cliente.
- Probar otros algoritmos como Gradient Boosting o Support Vector Machines (SVM).
- Incrementar el tamaño del conjunto de datos para mejorar el desempeño del modelo.


## Instrucciones para ejecutar el proyecto:
1. Clonar el repositorio
   ```
   git clone https://github.com/ErayFaSol/Sprint-8-introduccion-al-machine-learning
   cd Sprint-8-introduccion-al-machine-learning
   ```
2. Instalar las dependencias
   ```
   pip install -r requirements.txt
   ```
3. Ejecuta el script principal
   ``` 
   python src/main.py
   ```
4. La ejecucion creara un archivo llamado *reporte_final.html*


### Enlace al proyecto
[Introducción al Machine Learning](https://github.com/ErayFaSol/Sprint-8-introduccion-al-machine-learning)
