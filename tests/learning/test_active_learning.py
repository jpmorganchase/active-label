import numpy as np
import pandas as pd
import pytest
from modAL.models import ActiveLearner as ModALLearner
from sklearn.neighbors import KNeighborsClassifier

from active_label.annotations import Annotator
from active_label.learning import ActiveLearner


@pytest.fixture()
def features():
    return pd.DataFrame(np.linspace(0, 1, 100)[1:-1], columns=["X"])


@pytest.fixture()
def dataset(features):
    return {i: row.values.item() for i, row in features.iterrows()}


def test_classification(dataset, features):
    X = [[0], [1]]
    y = [0, 1]
    estimator = KNeighborsClassifier(n_neighbors=2)
    modal = ModALLearner(estimator, X_training=X, y_training=y)
    modal.bootstrap_init = None  # repr() bug
    learner = ActiveLearner(features, modal)

    annotator = Annotator(dataset, learner)
    annotator.pooled(dataset.keys())

    annotator.annotate()
    annotator.ignore()
    for i in range(6):
        if dataset[annotator.pending] < 0.5:
            annotator.no()
        else:
            annotator.yes()
    predict = modal.estimator.predict_proba([[0.25]])
    assert predict[0, 0] > 0.5
