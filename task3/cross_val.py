import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    indexes = np.arange(num_objects)
    fold_size = num_objects // num_folds
    remainder = num_objects % num_folds

    folds = []
    start_index = 0
    for i in range(num_folds):
        end_index = start_index + fold_size
        if i == num_folds - 1:
            end_index += remainder
        test_fold = indexes[start_index:end_index]
        train_fold = np.setdiff1d(indexes, test_fold)
        folds.append((train_fold, test_fold))
        start_index = end_index
    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    normalizer_name = None
    scores = {}
    for normalizer in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:
                    fold_scores = []
                    for train_idx, test_ind in folds:
                        X_train, x_test = X[train_idx], X[test_ind]
                        y_train, y_test = y[train_idx], y[test_ind]
                        normalizer_ob, normalizer_name = normalizer
                        if normalizer_ob is not None:
                            normalizer_ob.fit(X_train)
                            X_train = normalizer_ob.transform(X_train)
                            x_test = normalizer_ob.transform(x_test)
                        knn = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(x_test)
                        fold_scores.append(score_function(y_test, y_pred))
                    avg_score = np.mean(fold_scores)
                    scores[(normalizer_name, n_neighbors, metric, weight)] = avg_score
    return scores
