# Ejercicios para practicar

Todos los ejercicios usan el dataset [weatherAUS](https://www.kaggle.com/datasets/sandhyapalaniappan/rainfall-prediction-dataset-cleaned-weatheraus),
que contiene diez años de observaciones meteorológicas diarias de 49 estaciones de Australia. La variable a predecir
es `RainTomorrow`, que indica si llovió al día siguiente.

Se trata de una versión ya depurada del dataset original: **no tiene valores faltantes ni duplicados**. El trabajo de
preparación, entonces, no pasa por imputar sino por **decidir cómo tratar cada variable**.

---

1. **Preparación de los datos y primer modelo.**

   1. Cargue el dataset en un DataFrame de Pandas y verifique que efectivamente no haya valores faltantes.
   2. Separe las columnas en tres grupos: las **numéricas continuas** (temperaturas, humedad, presión, viento,
   nubosidad), las **categóricas** (`Location`, las tres direcciones de viento y `RainToday`) y las que conviene
   descartar o transformar aparte (`Date`). Observe que el target también viene como texto (`"Yes"` / `"No"`) y hay
   que codificarlo.
   3. Calcule la **tasa base** de lluvia. Con ese valor obtenga el Brier Score del modelo que predice siempre esa
   tasa, $\bar{y}(1-\bar{y})$. Va a ser el piso contra el cual comparar todo lo que entrene.
   4. Separe en entrenamiento y testeo (80 % – 20 %), estratificando por el target.
   5. Arme un `ColumnTransformer` que aplique `OneHotEncoder` a las categóricas y `StandardScaler` a las numéricas,
   y entrene una **regresión logística** dentro de un `Pipeline`.
   6. Repita el entrenamiento usando **solo las variables numéricas**, sin las categóricas. Compare AUC y Brier
   Score entre ambas versiones. ¿Cuánto aportó incorporar las categóricas?

2. **Discriminación vs. calibración.**

   1. Sobre el mismo split, entrene además un **Random Forest**.
   2. Obtenga las probabilidades predichas de ambos modelos y grafique sus **histogramas**. ¿Alguno de los dos evita
   los valores extremos? ¿Por qué le pasaría eso a un ensamble de árboles?
   3. Dibuje las **curvas de calibración** de ambos modelos en un mismo gráfico, junto con la diagonal. Acompáñelas
   con el histograma de probabilidades predichas: sin él no se puede distinguir una desviación real de un bin con
   pocas observaciones.
   4. Evalúe Accuracy, AUC, Brier Score y Log-loss para los dos modelos.
   5. Implemente el **Expected Calibration Error (ECE)** y calcúlelo para ambos. Scikit-learn no lo trae, pero son
   pocas líneas: agrupe las predicciones en bins y promedie $|\bar{y}_m - \bar{p}_m|$ ponderando por el tamaño de
   cada bin.
   6. Ahora responda: **¿cuál de los dos modelos está mejor calibrado?** Va a encontrar que el modelo con **mejor**
   Brier Score y mejor Log-loss es el que tiene **peor** ECE. Explique la aparente contradicción usando la
   descomposición del Brier Score en confiabilidad, resolución e incertidumbre. ¿Cuál de las dos métricas responde
   la pregunta que se hizo?

3. **Calibrar un modelo mal calibrado.**

   1. Entrene un **GaussianNB** usando **solo las variables numéricas**. Grafique el histograma de sus
   probabilidades y su curva de calibración.

      > Si intenta reusar el `ColumnTransformer` del ejercicio 1 se va a encontrar con un `TypeError`:
      > `OneHotEncoder` devuelve una matriz *sparse* y `GaussianNB` solo acepta datos densos. Se resuelve con
      > `OneHotEncoder(sparse_output=False)`, pero para este ejercicio alcanza con las numéricas.

   2. Compare su Brier Score contra el piso $\bar{y}(1-\bar{y})$ calculado en el ejercicio 1, y su AUC contra el de
   los modelos del ejercicio 2. Va a ver que discrimina razonablemente bien pero que su Brier apenas mejora al piso.
   ¿Cómo se explica esa combinación?
   3. Revise la matriz de correlación de las variables numéricas que usó. Busque los pares con correlación mayor a
   $0.9$ y explique, a partir de eso, por qué Naive Bayes produce probabilidades tan extremas.
   4. Calíbrelo con `CalibratedClassifierCV` usando `method="sigmoid"` y `method="isotonic"`, con `cv=5`.
   5. Grafique las tres curvas de calibración juntas y compare Brier Score y ECE. ¿Qué método funcionó mejor en este
   caso?
   6. Verifique qué pasó con el **AUC** y con el **accuracy** antes y después de calibrar. ¿Por qué el AUC casi no
   se mueve mientras que el accuracy sí puede cambiar?

4. **El error más común al calibrar.**

   1. Tome el Random Forest que entrenó en el ejercicio 2, ya ajustado sobre el conjunto de entrenamiento.
   2. Calíbrelo con regresión isotónica **usando los mismos datos de entrenamiento** con los que se ajustó el
   modelo. Para que el modelo base no se reentrene, envuélvalo en `FrozenEstimator` antes de pasarlo a
   `CalibratedClassifierCV`.
   3. Compare Brier Score, ECE y AUC contra el modelo sin calibrar. ¿Mejoró o empeoró?
   4. Repita la calibración, pero ahora con validación cruzada (`cv=5`) y el modelo **sin entrenar**. Compare las
   tres versiones.
   5. Para entender qué pasó, grafique la curva de calibración del Random Forest evaluada **sobre el conjunto de
   entrenamiento** y, al lado, la misma curva evaluada **sobre testeo**. Explique por qué un calibrador ajustado
   sobre la primera produce predicciones malas sobre la segunda.

5. **Decidir con probabilidades: elegir el umbral.**

   Un productor agrícola usa el modelo para decidir, cada noche, si cubre la cosecha. Cubrir cuesta **\$800** en mano
   de obra y materiales, llueva o no. Si no cubre y llueve, pierde **\$5000**.

   1. Plantee el costo esperado de cada decisión y despeje el **umbral óptimo** de probabilidad a partir del cual
   conviene cubrir. Verifique que no es $0.5$.
   2. Escriba una función que, dado un vector de probabilidades y las etiquetas reales, calcule el **costo total** de
   la temporada.
   3. Calcule ese costo para dos políticas triviales: cubrir siempre y no cubrir nunca.
   4. Calcule el costo usando el **GaussianNB sin calibrar** del ejercicio 3, primero con umbral $0.5$ y después con
   el umbral óptimo que despejó. ¿Cuánto se ahorra solo por mover el umbral?
   5. Repita con el modelo **calibrado**, usando el umbral óptimo. ¿Cuánto ahorra la calibración por encima del
   cambio de umbral?
   6. Discuta: si la decisión se tomara siempre con umbral $0.5$, ¿habría hecho alguna diferencia calibrar el
   modelo? ¿En qué situaciones la calibración es indistinta y en cuáles es decisiva?

6. **(Optativo) Calibración multiclase.**

   La columna `Rainfall` tiene los milímetros caídos en el día. Úsela para construir un target de tres clases:
   *sin lluvia*, *lluvia leve* y *lluvia fuerte*.

   1. Elija los cortes en milímetros y verifique la distribución resultante de las tres clases.
   2. Entrene un clasificador multiclase y obtenga la matriz de probabilidades con `predict_proba`. Verifique que
   cada fila suma 1.
   3. Grafique una curva de calibración **por clase**, usando el esquema *one-vs-rest*: para la clase $k$, compare la
   probabilidad predicha $p_k$ contra la frecuencia real de esa clase.
   4. Calibre el modelo con `CalibratedClassifierCV` y vuelva a graficar las tres curvas. Recuerde que internamente
   ajusta un calibrador binario por clase y después normaliza para que las probabilidades sumen 1.
   5. Discuta qué limitación tiene ese esquema de normalización.

---

**Nota sobre el split.** El dataset tiene una columna `Date` y los registros están ordenados en el tiempo. Separar en
entrenamiento y testeo al azar mezcla días futuros con pasados, lo cual en un problema de pronóstico es una fuga
temporal. Para los ejercicios de calibración el split aleatorio alcanza, pero si quiere hacerlo bien, pruebe separar
por fecha (por ejemplo, entrenar hasta 2015 y testear de 2016 en adelante) y observe si las conclusiones sobre
calibración cambian.
