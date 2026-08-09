import copy
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, train_test_split


def create_comemberships_matrix(y: np.ndarray, k: int) -> np.ndarray:
    """
    Create a co-memberships matrix from the results of a
    cluster model for the k-th cluster.

    Parameters
    ----------
    y : np.ndarray
        Clustering model output.
    k : int
        Number of the cluster.

    Returns
    -------
    np.ndarray
        2D array of booleans with the co-memberships matrix.
        An element matrix[i, j] of the matrix that equals True
        means that the element y[i] and y[j] are from the cluster k.
    """

    matrix = (y.reshape(-1, 1) == k) & (y.reshape(-1, 1) == y)
    # Remove comparizon between the same element
    np.fill_diagonal(matrix, 0)

    # Inefficient but easier to understand
    # matrix = np.zeros(shape=(y.shape[0], y.shape[0]))
    # for i in range(y.shape[0]-1):
    #    for j in range(i+1, y.shape[0]):
    #        if y[i] == y[j] == k:
    #            matrix[i, j] = 1
    #            matrix[j, i] = 1

    return matrix


def comprobate_with_model(
    model: Any, xx_test: np.ndarray, co_membership_matrix: np.ndarray
) -> np.ndarray:
    """
    Check which observations from the test dataset are in the
    same cluster in both models (test model and train model). The
    result is a new co-membership matrix without the intersection
    of both models.

    Parameters
    ----------
    model : Sklearn clustering object
        A trained sklearn clustering model with the train dataset.
    xx_test : np.ndarray
        Observations of the test dataset.
    co_membership_matrix : np.ndarray
        Co-memberships matrix from the test cluster model for
        the k-th cluster.

    Returns
    -------
    np.ndarray
        2D array of booleans with the co-memberships matrix of
        both models. An element matrix[i, j] of the matrix that
        equals True means that the elements xx_test[i] and xx_test[j]
        are from the cluster k, and in the train model, the two
        observations are in the same cluster.
    """

    # Calculate which cluster is the testing values with the training model
    y_pred = model.predict(xx_test)

    matrix_train_all_cluster = y_pred.reshape(-1, 1) == y_pred
    matrix_train_all_cluster *= co_membership_matrix

    # Inefficient but easier to understand
    # y_pred = model.predict(xx_test)

    # for i in range(co_membership_matrix.shape[0]-1):
    #    for j in range(i+1, co_membership_matrix.shape[1]):
    #        if co_membership_matrix[i, j] == 1:
    #            if y_pred[i] != y_pred[j]:
    #                co_membership_matrix[i, j] = 0
    #                co_membership_matrix[j, i] = 0
    #
    # matrix_train_all_cluster = co_membership_matrix

    return matrix_train_all_cluster


def prediction_strength_of_cluster(
    xx_test: np.ndarray, y_test: np.ndarray, training_model: Any, k: int
) -> float:
    """
    Calculate the prediction strength of a particular cluster
    (k-th cluster).
    The maximum value is 1, and the minimum value is 0. A value
    closer to 1 indicates consistent clustering (the cluster of
    the train and test dataset are in the same places).

    Parameters
    ----------
    xx_test : np.ndarray
        Observations of the test dataset.
    y_test : np.ndarray
        Cluster membership of the test dataset.
    training_model : Sklearn clustering object
        A trained sklearn clustering model with the train dataset.
    k : int
        Number of the cluster.

    Returns
    -------
    float
        Strength proportion for the k cluster in the test model.
    """

    # Obtain the number of element in the cluster
    cluster_length = np.sum(y_test == k)
    if cluster_length <= 1:
        return float("inf")

    # Obtain co-memberships matrix from the test model
    matrix = create_comemberships_matrix(y_test, k)

    # Obtain the co-memberships matrix from the intersection
    # between the test model and train model.
    matrix = comprobate_with_model(training_model, xx_test, matrix)

    # Count how many pairwise of the observation are still connected
    # from both models
    count = np.sum(matrix)
    proportion = count / (cluster_length * (cluster_length - 1))

    return float(proportion)


def calculate_prediction_strength(
    xx_test: np.ndarray,
    y_test: np.ndarray,
    training_model: Any,
    n_clusters: int,
    obtain_all_strengths: bool = False,
) -> float | list[float]:
    """
    Calculate the prediction strength for the number of clusters
    chosen.
    The maximum value is 1, and the minimum value is 0. A value
    closer to 1 indicates consistent clustering (all the clusters
    of the train and test dataset are in the same places).

    Parameters
    ----------
    xx_test : np.ndarray
        Observations of the test dataset.
    y_test : np.ndarray
        Cluster membership of the test dataset.
    training_model : Sklearn clustering object
        A trained sklearn clustering model with the train dataset.
    n_clusters : int
        Number of clusters in the y_test array.
    obtain_all_strengths : bool
        (Default value = False)
        If True, obtain the strength proportion for each cluster,
        else obtain the prediction strength (minimum strength
        proportion of all clusters).

    Returns
    -------
    float or list of float
        The prediction strength for the number of clusters chosen.
        Instead, if obtain_all_strengths is True, return a list
        with the strength proportion for each cluster.
    """

    prediction_strengths = [
        prediction_strength_of_cluster(xx_test, y_test, training_model, k)
        for k in range(n_clusters)
    ]

    if obtain_all_strengths:
        return prediction_strengths

    return float(np.min(prediction_strengths))


def prediction_strength_cross_validation(
    x_values: np.ndarray,
    clustering_model: Any,
    cross_validation_split: int,
    type_model: str = "clustering",
) -> tuple[float, float]:
    """
    Make a K-fold cross-validation for a clustering model using
    prediction strength as a metric.

    Parameters
    ----------
    x_values : np.ndarray
        Observations of the dataset.
    clustering_model : Sklearn clustering object
        A not trained sklearn clustering model.
    cross_validation_split : int
        Number of folds for the cross-validation.
    type_model : str
        Type of model to use for prediction strength. It can be a
        clustering model or a mixture model. Defaults to "clustering".

    Returns
    -------
    float
        Mean of the prediction strength across the folds.
    float
        Standard deviation of the prediction strength across the folds.
    """

    cv = KFold(n_splits=cross_validation_split, shuffle=True)
    results = np.zeros(cross_validation_split)

    for i, (train_index, test_index) in enumerate(cv.split(x_values)):
        xx_train = x_values[train_index]
        xx_test = x_values[test_index]

        results[i] = _obtain_metric_for_cv(
            xx_train, xx_test, clustering_model, type_model=type_model
        )

    return float(np.mean(results)), float(np.std(results))


def prediction_strength_half_split(
    x_values: np.ndarray, clustering_model: Any, repetitions: int
) -> tuple[float, float]:
    """
    Make a half-split cross-validation for a clustering model using
    prediction strength as a metric.

    Parameters
    ----------
    x_values : np.ndarray
        Observations of the dataset.
    clustering_model : Sklearn clustering object
        A not trained sklearn clustering model.
    repetitions : int
        Number of repetitions of the half-split.

    Returns
    -------
    float
        Mean of the prediction strength across the repetitions.
    float
        Standard deviation of the prediction strength across the repetitions.
    """
    results = np.zeros(repetitions)

    for i in range(repetitions):
        xx_train, xx_test = train_test_split(x_values, test_size=0.5)

        results[i] = _obtain_metric_for_cv(xx_train, xx_test, clustering_model)

    return float(np.mean(results)), float(np.std(results))


def _obtain_metric_for_cv(
    xx_train: np.ndarray,
    xx_test: np.ndarray,
    clustering_model: Any,
    type_model: str = "clustering",
) -> float:
    """
    Private function that calculates the prediction strength
    for a loop in the cross-validation process.

    Parameters
    ----------
    xx_train : np.ndarray
        Observations of the train dataset.
    xx_test : np.ndarray
        Observations of the test dataset.
    clustering_model : Sklearn clustering object
        A not trained sklearn clustering model.
    type_model : str
        Type of model to use for prediction strength. It can be a
        clustering model or a mixture model. Defaults to "clustering".

    Returns
    -------
    float
        Prediction strength for this train/test split.
    """

    # Create a new instance for each model
    train_model = copy.copy(clustering_model)
    test_model = copy.copy(clustering_model)

    train_model.fit(xx_train)
    test_model.fit(xx_test)

    y_test = test_model.predict(xx_test)

    if type_model == "clustering":
        clusters = test_model.n_clusters
    else:
        clusters = test_model.n_components

    result = calculate_prediction_strength(xx_test, y_test, train_model, clusters)
    assert isinstance(result, float)
    return result
