import random
from itertools import combinations

import pytest

from active_label.annotations import PairAnnotator
from active_label.learning.graph import PartialOrderLearner


def invariant(graph):
    def test(a, b):
        v = graph.get_edge_data(a, b)
        w = graph.get_edge_data(b, a)
        return v is not None and w is not None and v["value"] and w["value"]

    return not any([test(a, b) for a, b in graph.edges])


@pytest.fixture(scope="function")
def learner():
    return PartialOrderLearner()


def test_transitivity_triples_1(learner):
    # A < B < C
    learner.teach(("A", "B"), True)
    learner.teach(("B", "C"), True)
    assert len(learner.graph.nodes) == 3
    data = learner.graph.get_edge_data("A", "C")
    assert data["value"]


def test_transitivity_triples_2(learner):
    # A >= B >= C
    learner.teach(("A", "B"), False)
    learner.teach(("B", "C"), False)
    assert len(learner.graph.nodes) == 3
    data = learner.graph.get_edge_data("A", "C")
    assert not data["value"]


def test_sorting(learner):
    # A < B < C, D < E, F
    learner.teach(("A", "B"), True)
    learner.teach(("B", "C"), True)
    learner.teach(("D", "E"), True)
    learner.teach(("E", "F"), False)
    assert len(learner.graph.nodes) == 6
    sorts = list(learner.sorted_components())
    assert len(sorts) == 2
    sorts.sort(key=len)
    assert sorts[0] == ["D", "E"]
    assert sorts[1] == ["A", "B", "C"]
    sorts = list(learner.sorted_components(full=True))
    assert len(sorts) == 3
    sorts.sort(key=len)
    assert sorts[0] == ["F"]


def test_reverse_sorting(learner):
    # A >= B >= C
    learner.teach(("A", "B"), False)
    learner.teach(("B", "C"), False)
    assert len(learner.graph.nodes) == 3
    sorts = list(learner.sorted_components())
    assert len(sorts) == 0
    sorts = list(learner.sorted_components(reverse=True))
    assert len(sorts) == 1
    assert sorts[0] == ["A", "B", "C"]


def test_equality_1(learner):
    # A == B
    learner.teach(("A", "B"), False)
    learner.teach(("B", "A"), False)
    assert len(learner.graph.nodes) == 2
    assert learner.is_equal("A", "B")


def test_equality_2(learner):
    # A == B == C
    learner.teach(("A", "B"), False)
    learner.teach(("B", "A"), False)
    learner.teach(("B", "C"), False)
    learner.teach(("C", "B"), False)
    assert len(learner.graph.nodes) == 3
    assert learner.is_equal("A", "C")


def test_quad():
    nodes = set(range(0, 4))
    pairs = [(i, j) for i in nodes for j in nodes if i != j]
    init = pairs.pop()
    for tail in combinations(pairs, 4):
        path = [init] + list(tail)
        for n in range(2 ** len(path)):
            values = "{:04b}".format(n)
            learner = PartialOrderLearner()
            for p, x in zip(path, values):
                x = bool(x)
                learner.teach(p, x)
                assert invariant(learner.graph)


def check_order(learner, value_d):
    values = list(value_d.items())
    values.sort(key=lambda v: v[1])
    sorts = list(learner.sorted_components())
    assert len(sorts) == 1
    assert sorts[0] == [i[0] for i in values]


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_nodes, seed",
    [
        (2, 53),
        (3, 57),
        (8, 1234),
        (64, 583),
        (128, 42),
    ],
)
def test_simulation(learner, n_nodes, seed):
    random.seed(seed)
    value_d = {i: random.random() for i in range(0, n_nodes)}
    pairs = {(i, j) for i in value_d for j in value_d if i != j}
    annotator = PairAnnotator(dataset=value_d, learner=learner)
    annotator.pooled(pairs)
    annotator.annotate()
    while annotator.pending:
        i, j = annotator.pending
        if value_d[i] < value_d[j]:
            annotator.yes()
        else:
            annotator.no()
    check_order(learner, value_d)
    annotator.replay()
    check_order(learner, value_d)


@pytest.mark.slow
@pytest.mark.parametrize(
    "alpha, beta, n_nodes, n_steps",
    [
        (0, 0.5, 100, 200),
        (1, 0.5, 100, 200),
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
    pairs = {(i, j) for i in nodes for j in nodes if i != j}
    annotator = PairAnnotator(dataset={n: n for n in nodes}, learner=learner)
    annotator.pooled(pairs)
    annotator.annotate()
    for _ in range(1, n_steps):
        assert annotator.pending is None or not learner.graph.has_edge(*annotator.pending)
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
        assert invariant(learner.graph)
