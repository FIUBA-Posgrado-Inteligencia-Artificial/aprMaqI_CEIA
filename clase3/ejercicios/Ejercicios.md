# Ejercicios para prácticar


1. Usando el dataset `winequality-red.csv`, el cual consiste en datos de vinos rojos basados en datos físico-químicos y 
una métrica de calidad de vino. Más info en [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). 
Queremos predecir la calidad del vino usando los atributos físico-químicos del mismo.
   1. Lea el dataset como un DataFrame de Pandas. Realice un estudio de variables. Cómo se llaman y qué están midiendo 
   exactamente (vea la documentación del dataset). Además, analice qué tipo de variables (incluido el target) son, 
   cuál es el rango de estas variables y cómo se distribuyen (histograma). Además, realice una matriz de correlación, 
   ¿cuáles variables parecen estar correlacionadas? ¿Y con respecto a la calidad del vino?
   2. Realice si es necesaria la limpieza de datos y corrección de errores.
   3. Construya usando un SVM un modelo de regresión o clasificación multi-clase (según lo que considere más 
   apropiado) en el que se intente predecir la calidad del vino usando el nivel de alcohol.
      1. Realiza la separación entre el dataset de entrenamiento y testeo. Utilice 80 % - 20 %.
      2. Determine qué métrica se va a usar para evaluar la calidad del modelo (MSE, MAE, etc.)
      3. Entrene el modelo con el set de entrenamiento.
      4. Evalúe el modelo con la métrica de evaluación.

2. Continuando con el ejercicio de la clase 2 del dataset `UCI_Credit_Card.csv`. Incorpore al estudio de los modelos 
al menos un SVM de clasificación y repita las evaluaciones que se hicieron con los clasificadores. Discuta los 
resultados obtenidos.
