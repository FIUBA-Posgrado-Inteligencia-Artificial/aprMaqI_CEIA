import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class HeartDataset:
    """
    Clase para cargar, procesar y dividir el conjunto de datos Heart Disease

    Args:
        path (str): Ruta del archivo CSV que contiene los datos. Por defecto es "./Heart.csv".
        random_state (int): Semilla para la generación de números aleatorios. Por defecto es 42.
    """

    def __init__(self, path: str = "./Heart.csv", random_state: int = 42):
        """
        Inicializa una nueva instancia de HeartDataset.
        """
        self.path = path
        self.random_state = random_state
        df = pd.read_csv(path)
        df.drop_duplicates(inplace=True)

        # Hacemos variable dummy a las categóricas
        df_heart_dummies = self._create_dummies(df)

        # Spliteamos los datos y obtenemos el dataset estandarizado
        self.data_tuple = self._normalize(self._split_dataset(df_heart_dummies, force_floats=True))

        # Obtenemos las columnas
        self.columns_normalized = df_heart_dummies.columns

    def _create_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte variables categóricas en variables dummy.

        Args:
            df (DataFrame): DataFrame que contiene los datos originales.

        Returns:
            DataFrame: DataFrame con variables categóricas convertidas en variables dummy.
        """
        categorical_features = ["cp", "restecg", "slope", "ca", "thal"]

        return pd.get_dummies(data=df, columns=categorical_features, drop_first=True)

    def _split_dataset(self, df: pd.DataFrame, force_floats: bool = False) -> tuple:
        """
        Divide el conjunto de datos en conjuntos de entrenamiento y prueba.

        Args:
            df (DataFrame): DataFrame que contiene los datos a dividir.
            force_floats (bool): Booleano que indica si transforma los atributos en floats. Por defecto es False.

        Returns:
            tuple: Tupla que contiene los conjuntos de entrenamiento y prueba para características (X) y etiquetas (y).
        """
        X = df.drop(columns='target')
        if force_floats:
            X = X.astype(float)
        y = df['target'].astype("float")
        return train_test_split(X, y, test_size=0.3, random_state=self.random_state)

    def _normalize(self, splitted_tuple: tuple) -> tuple:
        """
        Normaliza las características del conjunto de datos.

        Args:
            splitted_tuple (tuple): Tupla que contiene los conjuntos de entrenamiento y prueba.

        Returns:
            tuple: Tupla que contiene características normalizadas para entrenamiento y prueba.
        """
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(splitted_tuple[0])
        X_test = sc_X.transform(splitted_tuple[1])

        return X_train, X_test, splitted_tuple[2], splitted_tuple[3]


def evaluate_classifier(name, y_test, y_pred, y_pred_proba) -> dict:
    """
    Evalúa el desempeño de un clasificador y retorna métricas.

    Args:
        name (str): Nombre del modelo.
        y_test (np.ndarray): Etiquetas verdaderas.
        y_pred (np.ndarray): Predicciones de clase.
        y_pred_proba (np.ndarray): Probabilidades de predicción.

    Returns:
        dict: Diccionario con métricas de evaluación del modelo.
    """

    dictionary = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_score": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return dictionary


def train_test_generic(name, cls, data_tuple):
    """
    Entrena un clasificador y evalúa su desempeño.

    Args:
        name (str): Nombre del modelo.
        cls: Clasificador a entrenar.
        data_tuple (tuple): Tupla con datos de entrenamiento y prueba.

    Returns:
        tuple: Clasificador entrenado y métricas de evaluación.
    """
    cls.fit(data_tuple[0], data_tuple[2])

    # Evaluamos el modelo
    y_pred = cls.predict(data_tuple[1])
    y_pred_proba = cls.predict_proba(data_tuple[1])
    metrics = evaluate_classifier(name, data_tuple[-1], y_pred, y_pred_proba[:, -1])

    return cls, metrics


def create_train_logistic_regression(dataset: HeartDataset):
    """
    Crea y entrena un modelo de Regresión Logística.

    Args:
        dataset (HeartDataset): Conjunto de datos.

    Returns:
        tuple: Modelo entrenado y métricas de evaluación.
    """

    cls = LogisticRegression(random_state=42, class_weight="balanced")

    cls_trained, metrics = train_test_generic("Regresión Logística", cls, dataset.data_tuple)

    return cls_trained, metrics


def create_train_svc(dataset: HeartDataset):
    """
    Crea y entrena un modelo de Support Vector Classifier (SVC).

    Args:
        dataset (HeartDataset): Conjunto de datos.

    Returns:
        tuple: Modelo entrenado y métricas de evaluación.
    """

    # Usamos el mejor modelo que vimos en la clase de SVM
    best_params = {'C': 5, 'kernel': 'linear'}
    cls = SVC(**best_params, probability=True, random_state=42)

    # Entrenamos el modelo y evaluamos el modelo
    cls_trained, metrics = train_test_generic("SVC", cls, dataset.data_tuple)

    return cls_trained, metrics


def create_train_tree(dataset: HeartDataset):
    """
    Crea y entrena un modelo de Árbol de Decisión.

    Args:
        dataset (HeartDataset): Conjunto de datos.

    Returns:
        tuple: Modelo entrenado y métricas de evaluación.
    """

    # Usamos el mejor modelo que vimos en la clase de árboles
    best_params = {'max_depth': 16,
                   'criterion': 'entropy',
                   'min_samples_split': 17,
                   'min_samples_leaf': 3}
    cls = DecisionTreeClassifier(**best_params, random_state=42)

    # Entrenamos el modelo y evaluamos el modelo
    cls_trained, metrics = train_test_generic("Tree", cls, dataset.data_tuple)

    return cls_trained, metrics


def create_train_knn(dataset: HeartDataset):
    """
    Crea y entrena un modelo de k-Nearest Neighbors (kNN).

    Args:
        dataset (HeartDataset): Conjunto de datos.

    Returns:
        tuple: Modelo entrenado y métricas de evaluación.
    """

    # Usamos el mejor modelo (se hizo una búsqueda de hiperparámetros previamente)
    best_params = {'n_neighbors': 35, 'p': 3.0, 'weights': 'distance'}
    cls = KNeighborsClassifier(**best_params)

    # Entrenamos el modelo y evaluamos el modelo
    cls_trained, metrics = train_test_generic("kNN", cls, dataset.data_tuple)

    return cls_trained, metrics


def obtain_best_threshold(dataset: HeartDataset, y_pred_proba: np.ndarray):
    """
     Obtiene el mejor umbral para la clasificación binaria.

     Args:
         dataset (HeartDataset): Conjunto de datos.
         y_pred_proba (np.ndarray): Probabilidades de predicción.

     Returns:
         float: Umbral óptimo.
     """

    # Calculamos la curva de ROC
    fpr, tpr, thresholds = roc_curve(dataset.data_tuple[-1], y_pred_proba[:, 1])

    # Calculamos el estadístico J de Youden para cada umbral
    youden_j = tpr - fpr

    # Buscamos el umbral optimo
    optimal_threshold = thresholds[np.argmax(youden_j)]

    return optimal_threshold
