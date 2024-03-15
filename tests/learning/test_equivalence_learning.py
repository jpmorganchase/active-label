import random
from itertools import combinations

import pytest

from active_label.annotations import PairAnnotator
from active_label.learning.graph import EquivalenceLearner


@pytest.fixture
def learner():
    return EquivalenceLearner()


def test_transitivity_triples_1(learner):
    learner.teach(("A", "B"), True)
    learner.teach(("B", "C"), True)
    assert len(learner.graph.nodes) == 3
    assert len(list(learner.equivalence_classes())) == 1
    data = learner.graph.get_edge_data("A", "C")
    assert data["value"]


def test_transitivity_triples_2(learner):
    learner.teach(("A", "B"), True)
    learner.teach(("B", "C"), False)
    assert len(learner.graph.nodes) == 3
    assert len(list(learner.equivalence_classes(full=True))) == 2
    assert len(list(learner.equivalence_classes())) == 1
    data = learner.graph.get_edge_data("A", "C")
    assert not data["value"]


def test_transitivity_triples_3(learner):
    learner.teach(("A", "B"), False)
    learner.teach(("B", "C"), True)
    assert len(learner.graph.nodes) == 3
    assert len(list(learner.equivalence_classes(full=True))) == 2
    assert len(list(learner.equivalence_classes())) == 1
    data = learner.graph.get_edge_data("A", "C")
    assert not data["value"]


def test_transitivity_triples_4(learner):
    learner.teach(("A", "B"), False)
    learner.teach(("B", "C"), False)
    assert len(learner.graph.nodes) == 3
    assert len(list(learner.equivalence_classes(full=True))) == 3
    assert len(list(learner.equivalence_classes())) == 0
    data = learner.graph.get_edge_data("A", "C")
    assert data is None


def test_transitivity_quads_1(learner):
    learner.teach(("A", "B"), True)
    learner.teach(("C", "D"), True)
    learner.teach(("A", "C"), True)
    assert len(learner.graph.nodes) == 4
    assert len(list(learner.equivalence_classes(full=True))) == 1
    assert len(list(learner.equivalence_classes())) == 1
    data = learner.graph.get_edge_data("B", "D")
    assert data["value"]


def test_transitivity_quads_2(learner):
    learner.teach(("A", "B"), True)
    learner.teach(("C", "D"), True)
    learner.teach(("A", "C"), False)
    assert len(learner.graph.nodes) == 4
    assert len(list(learner.equivalence_classes(full=True))) == 2
    assert len(list(learner.equivalence_classes())) == 2
    data = learner.graph.get_edge_data("B", "D")
    assert not data["value"]


def test_transitivity_quads_3(learner):
    learner.teach(("A", "B"), True)
    learner.teach(("C", "D"), False)
    learner.teach(("A", "C"), True)
    assert len(learner.graph.nodes) == 4
    assert len(list(learner.equivalence_classes(full=True))) == 2
    assert len(list(learner.equivalence_classes())) == 1
    data = learner.graph.get_edge_data("B", "D")
    assert not data["value"]


def test_transitivity_quads_4(learner):
    learner.teach(("A", "B"), True)
    learner.teach(("C", "D"), False)
    learner.teach(("A", "C"), False)
    assert len(learner.graph.nodes) == 4
    assert len(list(learner.equivalence_classes(full=True))) == 3
    assert len(list(learner.equivalence_classes())) == 1
    data = learner.graph.get_edge_data("B", "D")
    assert data is None


@pytest.mark.slow
@pytest.mark.parametrize(
    "alpha, beta, n_nodes, n_steps",
    [
        (1, 0.5, 100, 110),
        (0.1, 0.5, 4, 10),
        (0.6, 0.7, 10, 1000),
        (0.4, 0.7, 10, 1000),
        (0.5, 0.2, 10, 1000),
        (0.5, 0.8, 10, 1000),
    ],
)
def test_random_teaching(learner, alpha, beta, n_nodes, n_steps):
    random.seed(42)
    nodes = set(range(0, n_nodes))
    pairs = set(combinations(nodes, 2))
    annotator = PairAnnotator(dataset={n: n for n in nodes}, learner=learner)
    annotator.pooled(pairs)
    annotator.annotate()
    for _ in range(1, n_steps):
        if random.random() < alpha:
            if random.random() < beta:
                annotator.yes()
            else:
                annotator.no()
        else:
            annotator.back()
        if len(annotator.values) == 0:
            assert len(learner.graph.edges) == 0
        if annotator.pending is None:
            assert len(learner.graph.nodes) == len(nodes)
            assert len(learner.graph.edges) == len(pairs)
        assert 0 <= len(learner.graph.nodes) <= len(nodes)
        assert 0 <= len(learner.graph.edges) <= len(pairs)


def check_classification(learner, class_d):
    values = set(class_d.values())
    classes = list(learner.equivalence_classes(full=True))
    assert len(classes) == len(values)


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_nodes, n_classes, seed",
    [
        (2, 1, 53653),
        (2, 2, 3223),
        (3, 1, 5735),
        (8, 1, 34578),
        (8, 4, 32578),
        (8, 8, 34578),
        (64, 32, 58322),
        (128, 32, 42325),
        (128, 82, 456679),
    ],
)
def test_simulation(learner, n_nodes, n_classes, seed):
    random.seed(seed)
    class_d = {i: random.randint(1, n_classes) for i in range(0, n_nodes)}
    pairs = set(combinations(class_d.keys(), 2))
    annotator = PairAnnotator(dataset=class_d, learner=learner)
    annotator.pooled(pairs)
    annotator.annotate()
    while annotator.pending:
        i, j = annotator.pending
        if class_d[i] == class_d[j]:
            annotator.yes()
        else:
            annotator.no()
    check_classification(learner, class_d)
    annotator.replay()
    check_classification(learner, class_d)
