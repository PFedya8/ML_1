from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """ 
    sum = 0
    has_non_neg = False
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            sum += X[i][i]
            has_non_neg = True
    if has_non_neg:
        return sum
    else:
        return -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return sorted(x) == sorted(y) if len(x) == len(y) else False


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max_prod = -1
    for i in range(len(x) - 1):
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            max_prod = max(max_prod, x[i] * x[i + 1])
    return max_prod


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    return [
        [
            sum(pixel[i] * weights[i] for i in range(len(weights)))
            for pixel in row
        ]
        for row in image
    ]


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    if sum(count for value, count in x) != sum(count for value, count in y):
        return -1

    scalar_product = 0
    i, j = 0, 0
    while i < len(x) and j < len(y):
        common_count = min(x[i][1], y[j][1])
        scalar_product += x[i][0] * y[j][0] * common_count

        x[i][1] -= common_count
        y[j][1] -= common_count

        if x[i][1] == 0:
            i += 1
        if y[j][1] == 0:
            j += 1

    return scalar_product




def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    def dot(x, y):
        return sum(a * b for a, b in zip(x, y))

    def norm(x):
        return sum(a * a for a in x) ** 0.5

    result = []
    for x in X:
        row = []
        for y in Y:
            if norm(x) == 0 or norm(y) == 0:
                row.append(1.0)
            else:
                cos_sim = dot(x, y) / (norm(x) * norm(y))
                row.append(cos_sim)
        result.append(row)

    return result

