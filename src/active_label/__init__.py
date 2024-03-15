"""
Fast annotations and active learning in Jupyter notebooks.

.. seealso::
    modAL active learning framework https://modal-python.readthedocs.io
"""

try:
    from importlib import metadata
except ImportError:  # pragma: no cover
    # Python < 3.8
    import importlib_metadata as metadata  # pragma: no cover

from active_label.annotations import Annotator, PairAnnotator

#: Library version, e.g. 1.0.0, taken from Git tags
__version__ = metadata.version("active-label")

__all__ = [
    "Annotator",
    "PairAnnotator",
]
