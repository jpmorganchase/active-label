"""Annotation

This module provides a lightweight interactive annotator for Jupyter notebooks.

Usage
-----

If you don't have a model you can simply annotate a batch of examples like so:

.. code-block:: python

    from active_label.annotations import Annotator

    dataset = {'a': 'doc a', 'b': 'doc b'}

    annotator = Annotator(dataset)
    annotator.pooled(dataset.keys())
    annotator.annotate()

Which would present the simplistic annotation UI with 7 buttons below which there
are the question and progress display, which shows the number of labeled samples and
the size of the remaining pool followed the default value.

The first two buttons mark values True, or False respectively and removes
the pending ID from the pool. "Back" will return the last ID to the pool and removes
it value from the list of assigned values.

The "Accept" accepts the default value indicated by the value as the true value.
The default value is "ignore" if none is provided.

The annotator also listens to keyboard shortcuts for Yes = ↑, No = ↓, Accept = → and Back = ←,
but the mouse focus must be over the annotator display. Otherwise, the notebook receives
the key events.

We can the see the annotations as tuple of lists of (IDs, values):

.. code-block:: python

    all_annotations = annotator.annotations()
    no_annotations = annotator.annotations(False)
    yes_annotations = annotator.annotations(True)

To export the results into a Pandas object call to_series():

.. code-block:: python

    df['values'] = annotator.to_series()

Assuming 'df' is pd.DataFrame indexed by the same IDs that was given to the annotator.

NB. None should always be considered a missing value.
NNB. If an ID was in the pool but not in df.index that value will not be written into df.

Pairwise Comparisons
--------------------

Many annotation tasks are naturally handled by comparing pairs of instances. PairAnnotator and
GraphLearner are designed to facilitate these. PairAnnotator presents pairs of articles side-by-side.
The only difference of PairAnnotator to Annotator is that it expect 'pool' to be a a list or tuple
of at least length 2 and presents the two first articles in these tuples.

.. code-block:: python

    from itertools import combinations

    annotator = PairAnnotator(dataset, display=display_pair)
    pairs = set(combinations(df.index, 2))
    annotator.pooled(pairs)
    annotator.annotate()

After annotating the dataset you can get the connected components with:

.. code-block:: python

    df['cluster'] = None
    c = 0
    for component in learner.connected_components():
        df.loc[component, 'cluster'] = c
        c += 1

Persistence
-----------

It is recommended to split a large annotation task into multiple small pieces. For that
it is also a good idea to store intermediate annotation tasks on the filesystem.

.. code-block:: python

    annotator.to_file('annotator.json')

which can later be loaded with:

.. code-block:: python

    annotator = Annotator(print, dataset)
    annotator.from_file('annotator.json')

It should be noted that the display-function and the dictionary of the dataset are not
stored and must be loaded or reconstructed otherwise. The stored JSON file has the
attributes 'pool', 'values' and 'index'.

Annotation task results can be combined with:

.. code-block:: python

    annotator2 = Annotator(print, dataset)
    annotator2.pooled(dataset.keys())
    annotator2.annotate()

    annotator.append(annotator2)

or from a file with:

.. code-block:: python

    annotator2.to_file('annotator2.json')
    annotator.append("annotator2.json")

The annotation results will be concatenated and thus the same ID can have
multiple values in this case.

"""

import abc
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

try:
    from itertools import pairwise  # Python >= 3.10
except ImportError:
    from more_itertools import pairwise  # noqa: F401

import IPython.display
import ipywidgets as widgets
import jsonpickle
import pandas as pd
from ipyevents import Event

from active_label.learning import ActiveLearner, Learner, PassiveLearner

logger = logging.getLogger(__package__)


def _click(func):
    """Widgets expect a function the first argument of which is the widget activated"""
    return lambda _: func()


def _yesno(value):
    if value is None:
        return "Ignore"
    return "Yes" if value else "No"


class AnnotatorBase:
    """Base Annotator that defines the UI"""

    _out: widgets.Output = None

    def __init__(self):
        """Initialization of the annotator"""
        self._out = widgets.Output()

    def _on_keydown_event(self, event):
        key = event["key"]
        if key == "ArrowUp":
            self.yes()
        elif key == "ArrowDown":
            self.no()
        elif key == "ArrowRight":
            self.accept()
        elif key == "ArrowLeft":
            self.back()
        elif key == "Delete":
            self.ignore()
        elif key == " ":
            self.skip()
        elif key == "Enter":
            self.mark()

    def _create_widgets(self):
        yes = widgets.Button(description="Yes ↑")
        yes.on_click(_click(self.yes))
        no = widgets.Button(description="No ↓")
        no.on_click(_click(self.no))
        accept = widgets.Button(description="Accept →")
        accept.on_click(_click(self.accept))
        back = widgets.Button(description="Back ←")
        back.on_click(_click(self.back))
        ignore = widgets.Button(description="Ignore ⌦")
        ignore.on_click(_click(self.accept))
        skip = widgets.Button(description="Skip ⎵")
        skip.on_click(_click(self.skip))
        mark = widgets.Button(description="Mark ↵")
        mark.on_click(_click(self.mark))
        buttons = [yes, no, accept, back, ignore, skip, mark]
        box = widgets.VBox([widgets.HBox(buttons), self._out])
        event = Event(source=box, watched_events=["keydown"])
        event.on_dom_event(self._on_keydown_event)
        return box

    @abc.abstractmethod
    def refresh(self):
        """Refresh the display"""
        pass

    def annotate(self):
        """Start the annotation loop"""
        self.update(True)
        widget = self._create_widgets()
        self.refresh()
        return widget

    @abc.abstractmethod
    def yes(self):
        """Annotate the current example with the answer 'Yes'"""
        pass

    @abc.abstractmethod
    def no(self):
        """Annotate the current example with the answer 'No'"""
        pass

    @abc.abstractmethod
    def accept(self):
        """Accept the default value as the answer"""
        self.ignore()

    @abc.abstractmethod
    def ignore(self):
        """Ignore the current example and remove it from the pool"""
        pass

    @abc.abstractmethod
    def skip(self):
        """Ask a new example"""
        pass

    @abc.abstractmethod
    def back(self):
        """Go back to the previous example and revert the answer"""
        pass

    @abc.abstractmethod
    def mark(self):
        """Bookmark the current example"""
        pass

    @abc.abstractmethod
    def update(self, force: bool = False):
        """Update the example from the pool"""
        pass

    @abc.abstractmethod
    def to_dict(self) -> Dict:
        """Returns a dictionary of the annotator.
        NB. For efficient serialization purposes the documents is not included in this dictionary.

        :returns:   A dictionary representation of this annotator
        """
        pass

    @abc.abstractmethod
    def from_dict(self, data: Dict):
        """Read the values of the annotator from a dictionary.

        :param data:    A dictionary representation of the annotator
        """
        pass

    def to_file(self, path: Path):
        """Store the dictionary from to_dict() into a file in JSON format using jsonpickle.
        We are using jsonpickle because Python sets require a special encoder.

        :param  path:   path of a file to store
        """
        obj = self.to_dict()
        x = jsonpickle.encode(obj)
        with open(path, "w", encoding="utf8") as f:
            f.write(x)

    def from_file(self, path: Path):
        """Load the dictionary using from_dict() from a file in JSON format using jsonpickle.
        We are using jsonpickle because Python sets require a special encoder.
        NB. jsonpickle has the same security vulnerabilities as pickle and should only be used from trusted sources.

        :param  path:   path of a JSON file
        """
        with open(path, "r", encoding="utf8") as f:
            x = f.read()
        obj = jsonpickle.decode(x)
        self.from_dict(obj)


class Annotator(AnnotatorBase):
    """A light annotator for annotating documents in Jupyter notebooks"""

    pending = None

    def __init__(
        self,
        dataset: Dict[str, Any] = None,
        learner: Learner = None,
        display=IPython.display.display,
        question: str = None,
        defaults: Dict = None,
    ):
        """Initialize the annotator

        :param display:     A display function that is assumed to take a single argument compatible
                            with SNAP document and displaying this in context appropriate manner
        :param dataset:     A dictionary of data items by their ID
        :param learner:     Learner (defaults to NullLearner)
        """
        super().__init__()
        self.marked = set()
        self.pool = set()
        self.values = OrderedDict()
        self.learner = learner if learner is not None else PassiveLearner()
        self.display = display
        self.dataset = dataset
        self.question = question
        self.defaults = defaults if defaults is not None else {}

    def annotations(self, value=None) -> Tuple[List, List]:
        """Returns a generator of (id, value) pairs with None values ignored

        :param value:   If not None returns the indices of the annotations with 'value'
        """

        def _test(x):
            x = x[1]
            if x is None:
                return False
            if value is not None and x != value:
                return False
            return True

        index, values = zip(*filter(_test, self.values.items()))
        return list(index), list(values)

    def pooled(self, pool: Set):
        """Add documents to the pool to be reviewed

        :param pool:    a set of document ids
        """
        logger.debug(f"Adding {len(pool)} items to the pool")
        self.pool |= pool - set(self.values)

    def unpool(self, x):
        """Remove the given document from the pool

        :param x:    a document id
        """
        if x in self.pool:
            self.pool.remove(x)
        else:
            self.learner.forget(x)
        if x == self.pending:
            self._query()
            self.refresh()
        if x in self.values:
            del self.values[x]

    def back(self):
        """Go back to the previous example and revert the answer"""
        if self.values:
            self.pending, value = self.values.popitem()
            logger.debug(f"Going back to {self.pending}")
            self.pool.add(self.pending)
            self.learner.forget(self.pending)
            self.update(force=False)

    def _resolve(self, value: bool, force: bool):
        self._teach(value)
        self.update()

    def yes(self, force=False):
        """Annotate the current example with the answer 'Yes'"""
        return self._resolve(True, force)

    def no(self, force=False):
        """Annotate the current example with the answer 'No'"""
        return self._resolve(False, force)

    def accept(self, force=False):
        """Accept the default value as the answer"""
        value = self._get_default()
        self._resolve(value, force)

    def ignore(self):
        """Ignore the current example and remove it from the pool"""
        self.defaults[self.pending] = None
        self._teach(None)
        self.update()

    def skip(self):
        """Ask a new example"""
        self.update()

    def mark(self):
        """Bookmark the current example"""
        if self.pending is not None:
            if self.pending in self.marked:
                self.marked.remove(self.pending)
            else:
                self.marked.add(self.pending)
        self.refresh()

    def _get_default(self):
        default = self.defaults[self.pending] if self.pending in self.defaults else None
        if default is None and self.learner is not None and isinstance(self.learner, ActiveLearner):
            default = self.learner.predict(self.pending)
        return default

    def _progress(self):
        mark = "*" if self.pending in self.marked else ""
        default = self._get_default()
        default = _yesno(default)
        if self.question:
            print(self.question)
        print(f"{len(self.values)} / {len(self.pool)} [{default}] {mark}")

    def refresh(self):
        """Refresh the display"""
        with self._out:
            IPython.display.clear_output()
            self._progress()
            if self.pending:
                document = self.dataset[self.pending]
                self.display(document)

    def update(self, force=True):
        """Update the example from the pool"""
        self._query(force=force)
        self.refresh()

    def _query(self, force=True):
        if force or self.pending is None:
            self.pending = self.learner.query(self.pool)

    def _teach(self, value):
        if self.pending is not None:
            logger.debug(f"Teaching that {self.pending} has the value {value}")
            self.values[self.pending] = value
            self.defaults[self.pending] = value
            self.pool.remove(self.pending)
            if value is not None:
                self.learner.teach(self.pending, value)

    def replay(self):
        """Replay the annotations to the learner"""
        self.learner.reset()
        for i, value in self.values.items():
            self.learner.teach(i, value)

    def append(self, other: Union["Annotator", Path, str]):
        """Append the values from another Annotator, or one in file, to the results of this one.

        :param other:   Another Annotator or a path of a JSON file of one.
        """
        if not isinstance(other, Annotator):
            path = other
            other = Annotator()
            other.from_file(path)
        logger.debug(f"Appending {len(other.values)} items")
        self.values.update(other.values)
        self.pooled(other.pool)
        self.pending = None
        self.learner.fit(*self.annotations())

    def to_dict(self) -> Dict:
        """Returns a dictionary of the annotator.
        NB. For efficient serialization purposes the documents is not included in this dictionary.

        :returns:   A dictionary representation of this annotator
        """
        return {
            "question": self.question,
            "pool": self.pool,
            "pending": self.pending,
            "values": self.values,
            "marked": self.marked,
        }

    def from_dict(self, data: Dict):
        """Read the values of the annotator from a dictionary.

        :param data:    A dictionary representation of the annotator
        """
        self.question = data["question"]
        self.pool = data["pool"]
        self.pending = data["pending"]
        self.values = data["values"]
        self.marked = data["marked"]
        self.replay()

    def to_series(self, name: str = None, skipped: bool = False) -> pd.Series:
        """Convert the annotations into a Pandas Series

        :param name:        Name of the Series
        :param skipped:     Include skipped values
        :returns:           Series of values
        """
        if not self.values:
            return pd.Series(dtype="object")
        if skipped:
            return pd.Series(self.values.values(), name=name, index=self.values.keys())
        i, values = self.annotations()
        return pd.Series(values, index=i)


class PairAnnotator(Annotator):
    """Annotator for pairs of documents

    The usage is the same as Annotator but the 'pool' is assumed to be a set
    of tuple pairs of document IDs (the tuple can have more than 2 members
    but the annotator will only look at the first two.
    """

    def refresh(self, force=True):
        """Refresh the display"""
        with self._out:
            IPython.display.clear_output()
            self._progress()
            if self.pending:
                data1 = self.dataset[self.pending[0]]
                data2 = self.dataset[self.pending[1]]
                self.display(data1, data2)
