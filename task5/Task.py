import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.categories_ = {}

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        for column in X.columns:
            self.categories_[column] = sorted(X[column].unique())
        return self

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        output_rows = []
        for _, row in X.iterrows():
            row_out = []
            for column in X.columns:
                one_hot = np.zeros(len(self.categories_[column]), dtype=self.dtype)
                if row[column] in self.categories_[column]:
                    index = self.categories_[column].index(row[column])
                    one_hot[index] = 1
                row_out.extend(one_hot)
            output_rows.append(row_out)
        return np.array(output_rows)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.counters = {}

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        for column in X.columns:
            unique_values = X[column].unique()
            self.counters[column] = {}
            for value in unique_values:
                relevant_Y = Y[X[column] == value]
                success_rate = np.mean(relevant_Y)
                count_rate = len(relevant_Y) / len(Y)
                self.counters[column][value] = (success_rate, count_rate)
        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        transformed = []
        for _, row in X.iterrows():
            row_transformed = []
            for column in X.columns:
                value = row[column]
                success_rate, count_rate = self.counters[column].get(value, (0, 0))
                relattion = (success_rate + a) / (count_rate + b)
                row_transformed.extend([success_rate, count_rate, relattion])
            transformed.append(row_transformed)
        return np.array(transformed, dtype=self.dtype)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.fitted_encoders = []

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        np.random.seed(seed)
        self.fitted_encoders.clear()
        fold_indices = list(group_k_fold(len(X), self.n_folds, seed))

        for val_idx, train_idx in fold_indices:
            counter_encoder = SimpleCounterEncoder(self.dtype)
            counter_encoder.fit(X.iloc[train_idx], Y.iloc[train_idx])
            self.fitted_encoders.append((val_idx, counter_encoder))
        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        transformed = []
        for val_idx, counter_encoder in self.fitted_encoders:
            transformed_subset = counter_encoder.transform(X.iloc[val_idx], a, b)
            indexed_subset = np.hstack((transformed_subset, np.array(val_idx, dtype=int).reshape(-1, 1)))
            transformed.append(indexed_subset)
        transformed = np.vstack(transformed)
        transformed_data_sorted = transformed[np.argsort(transformed[:, -1])]
        return transformed_data_sorted[:, :-1]

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    x_unique = np.unique(x)
    x_weights = np.zeros(len(x_unique))
    for i, x_value in enumerate(x_unique):
        x_weights[i] = np.mean(y[x == x_value])

    return x_weights
