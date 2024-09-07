# Introducción al Machine Learning

Este proyecto se enfoca en desarrollar un modelo de clasificación para recomendar planes de telefonía móvil basados en el comportamiento de los clientes. Utiliza técnicas de machine learning supervisado para predecir cuál es el plan más adecuado para cada cliente en función de su perfil de uso.

## Tecnologías utilizadas
- Python
- pandas
- scikit-learn

## Objetivo
El objetivo del proyecto es utilizar datos históricos del comportamiento de los clientes para recomendar el plan de telefonía móvil más adecuado. Esto puede ayudar a mejorar la experiencia del cliente y aumentar la retención por parte de la empresa.

## Contexto
La empresa de telecomunicaciones desea optimizar las recomendaciones de planes de telefonía móvil para sus clientes actuales y potenciales. En lugar de ofrecer todos los planes de manera genérica, se busca personalizar la oferta en base a los hábitos de consumo de cada cliente, como el uso de datos, llamadas y mensajes de texto.

## Descripción del Proyecto
El análisis se basa en un conjunto de datos que contiene información sobre el uso de diferentes servicios por parte de los clientes de la empresa de telecomunicaciones. La variable objetivo es el plan de telefonía móvil que se puede recomendar a cada cliente, y las variables independientes son los patrones de uso, como la cantidad de datos consumidos, la duración de las llamadas, y la frecuencia de envío de mensajes de texto.

Se sigue el siguiente flujo de trabajo:
1. **Carga y exploración de datos**: Se cargan los datos, se inspeccionan las características principales y se visualizan para entender las correlaciones entre variables.
2. **Preprocesamiento de datos**: Se realizan tareas como limpieza de datos, imputación de valores faltantes, normalización y codificación de variables categóricas.
3. **División de los datos**: Se divide el conjunto de datos en conjunto de entrenamiento y de prueba para evaluar el rendimiento del modelo.
4. **Entrenamiento del modelo**: Se entrena un modelo de clasificación utilizando scikit-learn. Se probaron varios algoritmos, como regresión logística, árboles de decisión y random forest.
5. **Evaluación del modelo**: Se mide el desempeño del modelo utilizando métricas como la precisión, la matriz de confusión y el puntaje F1.

## Proceso

### Carga y Exploración de Datos
Los datos se cargan utilizando pandas y se realiza un análisis exploratorio para identificar distribuciones y correlaciones importantes. Se identifican patrones iniciales en los datos, como el consumo de datos y llamadas, que permiten segmentar a los clientes en diferentes categorías.

### Preprocesamiento
El conjunto de datos contiene algunas características categóricas que se codifican utilizando técnicas como One-Hot Encoding. Además, se tratan valores faltantes y se normalizan los datos para asegurar que las variables estén en la misma escala.

### Entrenamiento del Modelo
Se probaron diferentes modelos de clasificación, incluyendo:
- Regresión logística
- Árboles de decisión
- Random Forest

El mejor rendimiento lo obtuvo el modelo de **Random Forest**, que alcanzó una precisión del 82% en el conjunto de prueba.

### Evaluación del Modelo
El modelo se evaluó utilizando diferentes métricas:
- **Precisión**: 82%
- **Matriz de confusión**: Indicó un buen rendimiento general, aunque hubo algunas clases de planes de telefonía con menor precisión.
- **Puntaje F1**: 0.80

## Resultados
El modelo de clasificación logró recomendar correctamente el plan adecuado en el 82% de los casos. Esto implica que la empresa puede optimizar sus estrategias de marketing y personalización de planes, lo que podría aumentar la satisfacción del cliente y mejorar la retención.

## Conclusiones
El proyecto demostró que es posible utilizar técnicas de machine learning para personalizar los planes de telefonía móvil recomendados a los clientes. Sin embargo, hay espacio para mejorar en la precisión del modelo, por lo que futuras iteraciones podrían probar con más datos o algoritmos avanzados como redes neuronales.

### Futuras mejoras
- Incluir más características relacionadas con el perfil de cliente.
- Probar otros algoritmos como Gradient Boosting o Support Vector Machines (SVM).
- Incrementar el tamaño del conjunto de datos para mejorar el desempeño del modelo.

### Enlace al proyecto
[Introducción al Machine Learning](https://github.com/ErayFaSol/Sprint-8-introduccion-al-machine-learning)
