try:
    from itertools import pairwise  # Python >= 3.10
except ImportError:
    from more_itertools import pairwise  # noqa: F401
import pytest

from active_label import annotations, learning

DISPLAY = {}
dataset = {x: str(x) for x in "abcd"}
pool = set(dataset.keys())
pair_pool = set(pairwise(pool))
defaults = {i: True for i in dataset}


def display(item, *args):
    if args:
        item = tuple([item] + list(args))
    DISPLAY["displaying"] = item


@pytest.fixture()
def learner():
    return learning.PassiveLearner()


def get_annotator(monkeypatch):
    monkeypatch.setitem(DISPLAY, "displaying", None)
    a = annotations.Annotator(dataset, display=display, defaults=defaults)
    a.pooled(pool)
    return a


@pytest.fixture()
def annotator(monkeypatch):
    return get_annotator(monkeypatch)


@pytest.fixture()
def annotator2(monkeypatch):
    return get_annotator(monkeypatch)


@pytest.fixture()
def pair_annotator(monkeypatch):
    monkeypatch.setitem(DISPLAY, "displaying", None)
    a = annotations.PairAnnotator(dataset, display=display)
    a.pooled(pair_pool)
    return a


def test_unpool(annotator):
    i = annotator.pending
    j = list(pool - {i})[0]
    annotator.yes()
    annotator.unpool(i)
    assert j in annotator.pool
    annotator.unpool(j)
    assert j not in annotator.pool


def test_random_learner_empty(learner):
    assert learner.query(set()) is None


def test_random_learner(learner):
    assert learner.query({42}) == 42


def test_displaying(annotator):
    annotator.annotate()
    assert DISPLAY["displaying"] in dataset.values()


def test_mark(annotator):
    annotator.annotate()
    annotator.mark()
    assert annotator.pending in annotator.marked


def test_accept(annotator):
    annotator.annotate()
    assert len(annotator.values) == 0
    annotator.accept()
    assert len(annotator.values) == 1
    assert DISPLAY["displaying"] in dataset.values()
    assert list(annotator.values.values()) == [True]


def test_accept_back(annotator):
    annotator.annotate()
    i = annotator.pending
    assert len(annotator.values) == 0
    annotator.no()
    assert len(annotator.values) == 1
    annotator.back()
    assert len(annotator.values) == 0
    annotator.accept()
    assert len(annotator.values) == 1
    assert not annotator.values[i]
    assert DISPLAY["displaying"] in dataset.values()
    assert list(annotator.values.values()) == [False]


def test_yes(annotator):
    annotator.annotate()
    annotator.yes()
    assert DISPLAY["displaying"] in dataset.values()
    assert list(annotator.values.values()) == [True]


def test_no(annotator):
    annotator.annotate()
    annotator.no()
    assert DISPLAY["displaying"] in dataset.values()
    assert list(annotator.values.values()) == [False]


def test_end(annotator):
    annotator.annotate()
    annotator.yes()
    annotator.no()
    assert list(annotator.values.values()) == [True, False]
    annotator.no(None)


def test_skip(annotator):
    annotator.annotate()
    skipped = annotator.pending
    remaining = pool - {skipped}
    annotator.ignore()
    assert list(annotator.values.values()) == [None]
    assert annotator.pool == remaining


def test_skip_at_end(annotator):
    annotator.annotate()
    annotator.yes()
    annotator.no()
    annotator.ignore()
    assert list(annotator.values.values()) == [True, False, None]
    assert len(annotator.pool) == len(pool) - 3


def test_back(annotator):
    annotator.annotate()
    displaying = DISPLAY["displaying"]
    annotator.yes()
    annotator.back()
    assert DISPLAY["displaying"] == displaying
    assert len(annotator.values) == 0
    assert annotator.pool == pool


def test_back_at_start(annotator):
    annotator.annotate()
    displaying = DISPLAY["displaying"]
    annotator.back()
    assert DISPLAY["displaying"] == displaying
    assert len(annotator.values) == 0
    assert annotator.pool == pool


def test_yes_shortcut(annotator, mocker):
    annotator.annotate()
    spy = mocker.spy(annotator, "yes")
    annotator._on_keydown_event({"key": "ArrowUp"})
    spy.assert_called_once()


def test_no_shortcut(annotator, mocker):
    annotator.annotate()
    spy = mocker.spy(annotator, "no")
    annotator._on_keydown_event({"key": "ArrowDown"})
    spy.assert_called_once()


def test_back_shortcut(annotator, mocker):
    annotator.annotate()
    spy = mocker.spy(annotator, "back")
    annotator._on_keydown_event({"key": "ArrowLeft"})
    spy.assert_called_once()


def test_skip_shortcut(annotator, mocker):
    annotator.annotate()
    spy = mocker.spy(annotator, "accept")
    annotator._on_keydown_event({"key": "ArrowRight"})
    spy.assert_called_once()


def test_back_at_end(annotator):
    annotator.annotate()
    annotator.yes()
    displaying = DISPLAY["displaying"]
    annotator.no()
    annotator.back()
    assert DISPLAY["displaying"] == displaying
    assert list(annotator.values.values()) == [True]
    assert len(annotator.pool) == len(dataset) - 1


def test_append(annotator, annotator2):
    annotator.annotate()
    annotator.yes(None)
    annotator2.annotate()
    annotator2.no(None)
    annotator.append(annotator2)
    assert annotator.pool == pool - (set(annotator.values) | set(annotator2.values))


def test_append_from_file(annotator, annotator2, tmp_path):
    annotator.annotate()
    annotator.yes()
    annotator2.annotate()
    annotator2.no()
    annotator2.to_file(tmp_path / "test.json")
    annotator.append(tmp_path / "test.json")
    assert annotator.pool == pool - (set(annotator.values) | set(annotator2.values))


def test_files(annotator, annotator2, tmp_path):
    annotator.annotate()
    annotator.yes()
    annotator.to_file(tmp_path / "test.json")
    annotator2.from_file(tmp_path / "test.json")
    assert annotator.pool == annotator2.pool
    assert annotator.values == annotator2.values


def test_annotations(annotator):
    annotator.annotate()
    annotator.ignore()
    annotator.yes()
    index, values = annotator.annotations()
    assert values == [True]


def test_yes_annotations(annotator):
    annotator.annotate()
    i = annotator.pending
    annotator.yes()
    annotator.no()
    annotated, values = annotator.annotations(True)
    assert {i} == set(annotated)


def test_empty_series(annotator):
    annotator.annotate()
    series = annotator.to_series()
    assert len(series) == 0


def test_series(annotator):
    annotator.annotate()
    for i in range(len(dataset)):
        if i % 2:
            annotator.yes()
        else:
            annotator.no()
    series = annotator.to_series()
    assert pool == set(series.index)


def test_pair(pair_annotator):
    pair_annotator.annotate()
    assert pair_annotator.pending in pair_pool
