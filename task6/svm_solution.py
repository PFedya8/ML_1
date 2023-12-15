import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    gamma_value = 1 / (np.shape(train_features)[0] * train_features.var())

    svc_model = SVC(kernel='rbf', C=80000, gamma=gamma_value)
    model_pipeline = make_pipeline(StandardScaler(), svc_model)

    model_pipeline.fit(train_features, train_target)

    return model_pipeline.predict(test_features)
