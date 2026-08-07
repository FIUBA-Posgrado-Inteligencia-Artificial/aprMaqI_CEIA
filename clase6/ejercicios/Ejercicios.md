# Ejercicios para practicar

1. Use el dataset [Iris Species](https://archive.ics.uci.edu/dataset/53/iris). El cual consiste en 50 muestras de flores de tres especies distintas (Iris-setosa, Iris-versicolor y Iris-virginica) con diferentes características (Largo del sépalo, ancho del sépalo, largo del pétalo y ancho del pétalo). En este caso, sabemos a qué clase pertenece cada planta, pero podemos con ello evaluar métodos de agrupación.
   1. Implemente 1 modelo usando el algoritmo de K-means.
      1. Con el método del codo, determine el número de clusters. Cuanto clusters se eligió, es igual a la cantidad de especies de planta? Hay una correlación (por ejemplo, supongamos que quedan 4 clusters, en los cuales 2 corresponden perfectamente a un tipo de planta y las otras dos a la otras dos categorías respectivamente)? 
      2. Separe un 20% de los datos del dataset y guardelos.  
      3. Arme el modelo con el número óptimo de clúster usando el 80% de los datos restantes. Con los cluster obtenidos, determine a que clase de planta pertenece mediante algún método de mayoría. 
      4. Clasifique al 20% de los datos, usando el modelo de agrupamiento y el método de mayoría elegida. ¿Qué resultado obtuvo? (Elija la metrica de evaluación que mejor considere). ¿Qué tan correcto fue armando agrupamientos de datos con respecto a la clase que dan origen a los datos?
      5. Compare el procedimiento que realizó con KNN. ¿Que tán similar es al clasificador en cuanto al procedimiento?
   2. Implemente 1 modelo usando el algoritmo de GMM:
      1. Obtenga el número de óptimo de clusters usando Fuerza de predicción
      2. Separe un 20% de los datos del dataset y guardelos.  
      3. Arme el modelo con el número óptimo de clúster usando el 80% de los datos restantes. Con los cluster obtenidos, determine a que clase de planta. 
      4. Clasifique al 20% de los datos, usando el modelo de agrupamiento y el método de mayoría elegida. ¿Qué resultado obtuvo? (Elija la metrica de evaluación que mejor considere). ¿Qué tan correcto fue armando agrupamientos de datos con respecto a la clase que dan origen a los datos? 
   3. Implemente 1 modelo usando el algoritmo de Hierarchical clustering:
      1. Grafique el dendrograma. 
      2. Obtenga el número de óptimo de clusters usando la técnica de preferencia. 
      3. Separe un 20% de los datos del dataset y guardelos.  
      4. Arme el modelo con el número óptimo de clúster usando el 80% de los datos restantes. Con los cluster obtenidos, determine a que clase de planta. 
      5. Clasifique al 20% de los datos, usando el modelo de agrupamiento y el método de mayoría elegida. ¿Qué resultado obtuvo? (Elija la metrica de evaluación que mejor considere). ¿Qué tan correcto fue armando agrupamientos de datos con respecto a la clase que dan origen a los datos? 
      6. Compare con el resultado de Knn, K-means y de GMM. ¿Cómo fue el rendimiento de los diferentes modelos? 