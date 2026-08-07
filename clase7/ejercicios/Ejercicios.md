# Ejercicios para practicar

1. Use el dataset [weatherAUS](https://www.kaggle.com/datasets/sandhyapalaniappan/rainfall-prediction-dataset-cleaned-weatheraus). El dataset contiene información climática diaria y una variable binaria `RainTomorrow` indicando si llovió al día siguiente.

   1. Cargue el dataset en un DataFrame de Pandas, seleccione algunas variables relevantes (temperatura, humedad, presión, etc.) y realice una limpieza básica de datos faltantes.
   2. Separe en entrenamiento y testeo (80 % – 20 %).
   3. Entrene dos modelos de clasificación binaria: **Logistic Regression** y **Random Forest**.
   4. Obtenga las probabilidades predichas y grafique los **histogramas** de estas probabilidades para ambos modelos.
   5. Dibuje las **curvas de calibración** usando `CalibrationDisplay`.
   6. Evalúe Accuracy, Brier Score y Log-loss, y discuta brevemente cuál modelo parece estar mejor calibrado.

2. Usando el mismo dataset `weatherAUS`:

   1. Entrene un modelo **GaussianNB** con las mismas variables.
   2. Grafique el histograma de probabilidades y la curva de calibración del modelo sin calibrar.
   3. Calibre el modelo usando `CalibratedClassifierCV` con los métodos `"sigmoid"` e `"isotonic"`.
   4. Dibuje las curvas de calibración de ambos modelos calibrados y compare el Brier Score con el modelo original.
   5. Discuta brevemente cuál método de calibración funcionó mejor y si el Accuracy cambió luego de calibrar.
