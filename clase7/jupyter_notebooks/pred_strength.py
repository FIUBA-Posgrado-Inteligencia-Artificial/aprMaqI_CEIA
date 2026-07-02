import copy
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, train_test_split


def create_comemberships_matrix(y: np.array, k: int | float):
    """
    Create a co-memberships matrix from the results of a
    cluster model for the k-th cluster.

    Parameters
    ----------
    y : np.array
        Clustering model output.
    k : int or float
        Number of the cluster.

    Returns
    -------
    np.array
        2D array of booleans with the co-memberships matrix.
        An element matrix[i, j] of the matrix equal True means
        that the element y[i] and y[j] are from the cluster k.
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
    model: Any, xx_test: np.array, co_membership_matrix: np.array
):
    """
    Comprobate which observations from the test dataset are in the
    same cluster in both models (test model and train model). The
    result is a new co-membership matrix without the intersection
    of both models.

    Parameters
    ----------
    model : Sklearn clustering object
        A trained sklearn clustering model with the train dataset.
    X_test : np.array
        Observations of the test dataset.
    co_membership_matrix : np.array
        Co-memberships matrix from the test cluster model for
        the k-th cluster.

    Returns
    -------
    np.array
        2D array of booleans with the co-memberships matrix of
        both models. An element matrix[i, j] of the matrix
        equals True means that the elements X_test[i] and X_test[j]
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
    xx_test: np.array, y_test: np.array, training_model: Any, k: int | float
):
    """
    Calculate the prediction strength of a particular cluster
    (k-th cluster).
    The maximum value is 1, and the minimum value is 0. A value
    closer to 1 indicates consistent clustering (the cluster of
    the train and test dataset are in the same places).

    Parameters
    ----------
    X_test :  np.array
        Observations of the test dataset.
    y_test : np.array
        Cluster membership of the test dataset.
    training_model : Sklearn clustering object
        A trained sklearn clustering model with the train dataset.
    k : int | float
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

    return proportion


def calculate_prediction_strength(
    xx_test: np.array,
    y_test: np.array,
    training_model: Any,
    n_clusters: int,
    obtain_all_strengths: bool = False,
):
    """
    Calculate the prediction strength for the number of clusters
    chosen.
    The maximum value is 1, and the minimum value is 0. A value
    closer to 1 indicates consistent clustering (all the clusters
    of the train and test dataset are in the same places).

    Parameters
    ----------
    X_test : np.array
        Observations of the test dataset.
    y_test : np.array
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
    float or list
        The prediction strength for the number of clusters chosen.
        Instead, if obtain_all_strengths is True, return a list
        with the strength proportion for each cluster.
    """

    prediction_strengths = [
        prediction_strength_of_cluster(xx_test, y_test, training_model, k)
        for k in range(n_clusters)
    ]

    if not obtain_all_strengths:
        prediction_strengths = np.min(prediction_strengths)

    return prediction_strengths


def prediction_strength_cross_validation(
    x_values: np.array,
    clustering_model: Any,
    cross_validation_split: int,
    type_model: str = "clustering",
):
    """
    Make a K-fold cross-validation for a clustering model using
    prediction strength as a metric.

    Parameters
    ----------
    x_values :  np.array
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
    float, float
        Mean and standard deviation of the prediction strength.
    """

    cv = KFold(n_splits=cross_validation_split, shuffle=True)
    results = np.zeros(cross_validation_split)

    for i, (train_index, test_index) in enumerate(cv.split(x_values)):
        xx_train = x_values[train_index]
        xx_test = x_values[test_index]

        results[i] = _obtain_metric_for_cv(
            xx_train, xx_test, clustering_model, type_model=type_model
        )

    return np.mean(results), np.std(results)


def prediction_strength_half_split(
    x_values: np.array, clustering_model: Any, repetitions: int
):
    """
    Make a half-split cross-validation for a clustering model using
    prediction strength as a metric.

    Parameters
    ----------
    X :  np.array
        Observations of the dataset.
    clustering_model : Sklearn clustering object
        A not trained sklearn clustering model.
    repetitions : int
        Number of repetitions of the half-split.

    Returns
    -------
    float, float
        Mean and standard deviation of the prediction strength
        of the repetitions.
    """
    results = np.zeros(repetitions)

    for i in range(repetitions):
        xx_train, xx_test = train_test_split(x_values, test_size=0.5)

        results[i] = _obtain_metric_for_cv(xx_train, xx_test, clustering_model)

    return np.mean(results), np.std(results)


def _obtain_metric_for_cv(
    xx_train: np.array,
    xx_test: np.array,
    clustering_model: Any,
    type_model: str = "clustering",
):
    """Private function that calculates the prediction strength
    for a loop in the cross-validation process.
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

    return calculate_prediction_strength(xx_test, y_test, train_model, clusters)
