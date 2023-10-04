import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    diag = np.diag(X)
    non_neg = diag[diag >= 0]
    return non_neg.sum() if non_neg.size > 0 else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    products = x[:-1] * x[1:]
    mod_products = products[(x[:-1] % 3 == 0) | (x[1:] % 3 == 0)]
    return mod_products.max() if mod_products.size > 0 else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.tensordot(image, weights, axes=([-1], [0]))


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_decoded = np.repeat(x[:, 0], x[:, 1])
    y_decoded = np.repeat(y[:, 0], y[:, 1])

    if x_decoded.shape[0] != y_decoded.shape[0]:
        return -1

    return np.sum(x_decoded * y_decoded)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True).T

    dot_products = X @ Y.T

    denominator = X_norm * Y_norm
    valid_denominator = denominator != 0

    cosines = np.zeros_like(dot_products)
    cosines[valid_denominator] = dot_products[valid_denominator] / denominator[valid_denominator]
    if np.any(X_norm == 0) or np.any(Y_norm == 0):
        cosines[np.logical_or(X_norm == 0, Y_norm == 0)] = 1

    return cosines