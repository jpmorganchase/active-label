"""Active Learning

This module provides a lightweight wrapper for ModAL's learner so that it can be used
in conjunction with an interactive annotator in Jupyter notebooks.

Active Learning
---------------

The package also supports active learning with ModAL
(https://modal-python.readthedocs.io/en/latest/content/models/ActiveLearner.html)
We have facade classes that provide light wrappers to ModAL ActiveLearner. We do this
because the annotator needs the documents to display but the active learner needs the
vectorized features of the document.

The ActiveLearner needs an estimator which is required to follow the model in self.learner follows
Scikit-Learn estimator, https://scikit-learn.org/stable/developers/develop.html, contract with a
function fit(data, targets). The operation re-initializes the model and fits it with all current values
not just the latest one.

.. code-block:: python

    estimator = ...
    modal = modal.models.ActiveLearner(estimator)
    learner = ActiveLearner(features, modal)

There are 2 Learner subclasses: PassiveLearner (the default learner for Annotator) and ActiveLearner.
PassiveLearner only picks the first sample from the pool and fit() and teach() are NOPs. ActiveLearner
also has a batch-mode in which teach() is NOP. Batch-mode is useful in case where the estimator does not
support incremental fitting and fit is slow.
"""

import abc
import logging
from copy import copy
from typing import Any, List, Set

import numpy as np
import pandas as pd
from modAL.models import ActiveLearner as ModALLearner

logger = logging.getLogger(__package__)


class Learner:
    """Super class for a learner"""

    def __init__(self, features: pd.DataFrame = None):
        """Initialize

        :param features:    A pandas DataFrame of vectorized features
        """
        self.features = features

    @abc.abstractmethod
    def query(self, pool: Set) -> Any:
        """Query for the next item to present

        :param pool:    set of IDs to choose from
        :returns:       ID of a document
        """
        pass

    def teach(self, i: str, value: Any) -> None:
        """Teach the model that the ID with i has a value

        :param i:       ID of a document
        :param value:   Value to teach
        """
        pass

    def forget(self, i: str) -> None:
        """Tell the model to forget the value of ID i

        :param i:       ID of a document
        """
        pass

    def reset(self):
        """Reset the model to initial state"""
        pass

    def fit(self, index: List, values: List) -> None:
        """Fit the model. Assumes that the model in self.learner follows
        Scikit-Learn estimator contract with a function fit(data, targets).
        The operation re-initializes the model and fits it with all current values
        not just the latest one.

        :param index:   list of document indices
        :param values:  list of values
        """
        pass


class PassiveLearner(Learner):
    """Passive learner that doesn't implement teaching or fitting"""

    def query(self, pool: Set) -> Any:
        """Query for the next item to present

        The passive learner only chooses the first from the pool

        :param pool:    set of IDs to choose from
        :returns:       ID of a document
        """
        if not pool:
            return None
        return list(pool)[0]


class ActiveLearner(Learner):
    """Wrapper for ModAL learner"""

    def __init__(self, features: pd.DataFrame, modal: ModALLearner, batch=False):
        """Initialization

        :param features:    set of IDs to choose from
        :param modal:       ModAL active learner
        :param batch:       if batch-mode is set teaching is disabled
        """
        super().__init__(features)
        self.learner = modal
        self.X_initial = copy(modal.X_training)
        self.y_initial = copy(modal.y_training)
        self.learned = {}
        self.batch = batch

    def predict(self, i):
        """Use the model to predict the result of a data point

        :param i:       id of a data point
        """
        return self.learner.predict(self.features.loc[i].values.reshape(1, -1)).take(0)

    def query(self, pool: Set) -> Any:
        """Query for the next item to present

        :param pool:    set of IDs to choose from
        :returns:       ID of a document
        """
        features = self.features.loc[list(pool)]
        i, _ = self.learner.query(features)
        i = i[0]
        return features.index[i]

    def teach(self, i: str, value: Any) -> None:
        """Teach the model that the ID with i has a value

        :param i:       ID of a document
        :param value:   Value to teach
        """
        if not self.batch:
            self.learned[i] = value
            x = self.features.loc[i].values.reshape(1, -1)
            y = np.array([value])
            self.learner.teach(x, y)

    def forget(self, i: str) -> None:
        """Tell the model to forget the value of ID i

        :param i:       ID of a document
        """
        if i in self.learned:
            self.learned.pop(i)
            self.refresh()

    def refresh(self):
        """Refit the model with the already labeled and initial data"""
        index = list(self.learned.keys())
        values = list(self.learned.values())
        features = self.features.loc[index].values
        features = np.vstack([self.X_initial, features])
        values = np.array(values)
        values = np.hstack([self.y_initial, values])
        self.learner.fit(features, values)

    def fit(self, index: List, values: List) -> None:
        """Fit the model.

        :param index:   list of document indices
        :param values:  list of values
        """
        assert len(index) == len(values)
        self.learned = dict(zip(index, values))
        self.refresh()
