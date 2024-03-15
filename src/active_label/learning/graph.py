"""Graph Learning

The graph learners maintain a consistent graph struct so that inferred relations
are automatically added. For example in a classification task we label pairs
of examples with the semantics that the presented pairs belong to the same class.
This relation is an 'equivalence relation' and thus if we label 'A' and 'B' in
the same class and 'B' and 'C' in the same class we can infer that 'A' and 'C'
also belong to the same class without explicitly reviewing that pair.

Usage
-----

.. code-block:: python
    from itertools import combinations

    learner = EquivalenceLearner()
    annotator = PairAnnotator(dataset, display=display_pair, learner=learner)
    pairs = set(combinations(df.index, 2))
    annotator.pooled(pairs)
    annotator.annotate()

"""

import abc
import logging
from collections import defaultdict
from typing import Any, Generator, List, Set, Tuple

import networkx as nx
import pandas as pd

from active_label.learning import Learner

logger = logging.getLogger(__package__)


def _true_edge(graph: nx.Graph, e: Tuple):
    v = graph.get_edge_data(*e)
    return v is not None and v["value"]


def _false_edge(graph: nx.Graph, e: Tuple):
    v = graph.get_edge_data(*e)
    return v is not None and not v["value"]


class GraphLearner(Learner):
    """Base class for graph learner"""

    graph: nx.Graph

    def __init__(self, features: pd.DataFrame = None):
        super().__init__(features)
        self.induced_by = defaultdict(set)

    def reset(self):
        """Reset the model to initial state"""
        self.induced_by = defaultdict(set)

    def query(self, pool: Set) -> Any:
        """Query for the next item to present

        The passive learner only chooses the first from the pool

        :param pool:    set of IDs to choose from
        :returns:       ID of a document
        """
        for e in pool:
            a, b = e
            if not self.graph.has_edge(a, b):
                return e
        return None

    def _add_edge(self, edge, a, b, value):
        if self.graph.has_edge(a, b):
            data = self.graph.get_edge_data(a, b)
            assert data["value"] == value
        else:
            self.graph.add_edge(a, b, value=value)
            self.induced_by[edge].add((a, b))

    def forget(self, edge: Tuple) -> None:
        """Tell the model to forget the value of pair

        :param edge:    Pair of IDs of documents
        """
        if edge in self.induced_by:
            for i, j in self.induced_by[edge]:
                self.graph.remove_edge(i, j)
            del self.induced_by[edge]

    @abc.abstractmethod
    def is_equal(self, a, b):
        """Check if items 'a' and 'b' are equal

        :param a:   item ID
        :param b:   item ID
        """
        pass

    def equivalence_classes(self, full: bool = False) -> Generator:
        """Return the connected components generator of the graph. Basically
        relays to the networkx function connected_components() of the positive
        labeled sub-graph.

        :param full:    if True treat missing values for edges as False
        :returns:       generator of the connected components
        """
        G = nx.Graph()
        if full:
            G.add_nodes_from(self.graph)
        G.add_edges_from([(a, b) for a, b in self.graph.edges if self.is_equal(a, b)])
        return nx.connected_components(G)


class EquivalenceLearner(GraphLearner):
    """Passive graph learner that doesn't implement teaching or fitting
    for learning an equivalence relation.

    A binary relation on a set X is said to be an equivalence relation,
    if and only if it is reflexive, symmetric and transitive. i.e.:

    Reflexivity: a = a
    Symmetry: a = b iff b = a
    Transitivity: if a = b and b = c then a = c.
    """

    def __init__(self, features: pd.DataFrame = None):
        super().__init__(features)
        self.graph = nx.Graph()

    def reset(self):
        """Reset the model to initial state"""
        super().reset()
        self.graph = nx.Graph()

    def is_equal(self, a, b) -> bool:
        """Check if items 'a' and 'b' are equal

        :param a:   item ID
        :param b:   item ID
        """
        return _true_edge(self.graph, (a, b))

    def _components(self, node):
        members = defaultdict(set)
        members[True].add(node)
        if node in self.graph.nodes:
            for neighbor in self.graph.neighbors(node):
                value = self.graph.get_edge_data(node, neighbor)["value"]
                members[value].add(neighbor)
        return members

    def teach(self, pair: Tuple, value: bool) -> None:
        """Teach the model that the ID with pair has a value

        :param pair:    Pair of IDs of documents
        :param value:   Value to teach
        """
        a, b = pair
        if self.graph.has_edge(a, b):
            return
        a_classes = self._components(a)
        b_classes = self._components(b)
        for i in a_classes[True]:
            for j in b_classes[True]:
                self._add_edge(pair, i, j, value)
        if value:
            for i in a_classes[True]:
                for j in b_classes[False]:
                    self._add_edge(pair, i, j, False)
            for i in a_classes[False]:
                for j in b_classes[True]:
                    self._add_edge(pair, i, j, False)


class PartialOrderLearner(GraphLearner):
    """Learner for (strict) partial order relation for presented pairs

    An irreflexive, strong, or strict partial order is a homogeneous relation < on a
    set P that is irreflexive, asymmetric, and transitive; i.e., it satisfies the following
    conditions for all a,b,c in P:

    Irreflexivity: ¬ a < a, i.e. no element is related to itself (also called anti-reflexive).
    Asymmetry: if a < b then ¬ b < a (a ≥ b).
    Transitivity: if a < b and b < c then a < c.
    """

    def __init__(self, features: pd.DataFrame = None):
        super().__init__(features)
        self.graph = nx.DiGraph()

    def reset(self):
        """Reset the model to initial state"""
        super().reset()
        self.graph = nx.DiGraph()

    def is_equal(self, a, b):
        """Check if items 'a' and 'b' are equal. This means that
        both '¬ a < b' and '¬ b < a'

        :param a:   item ID
        :param b:   item ID
        """
        v = self.graph.get_edge_data(a, b)
        w = self.graph.get_edge_data(b, a)
        return v is not None and w is not None and not v["value"] and not w["value"]

    def _set_value(self, edge, a, b, value: bool):
        if a != b:
            self._add_edge(edge, a, b, value)
            if value:
                # asymmetry
                self._add_edge(edge, b, a, False)

    def all_topological_sorts(self, group: List, reverse: bool = False) -> Generator:
        """Returns a generator of all topological sorts of a sub-graph of the graph.

        A topological sort is a non-unique permutation of the nodes such that an edge
        from u to v implies that u appears before v in the topological sort order.

        :param group:   subset of the nodes of the graph
        :param reverse: if True return reversed sort
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(group)
        subgraph = self.graph.subgraph(group)
        test = _false_edge if reverse else _true_edge
        graph.add_edges_from([e for e in subgraph.edges if test(subgraph, e)])
        return nx.all_topological_sorts(graph)

    def sorted_components(self, full: bool = False, reverse: bool = False) -> Generator:
        """Returns a generator of topological sorts of the connected components of the graph.
        Returns the first sorting. It is possible that the order is ambiguous, e.g.:
        A < B and A < C, but without explicitly stating the order of B and C there are two
        possible sorts: A, B, C and A, C, B.

        :param full:        if True include all nodes
        :param reverse:     if True consider negated edges (NB. not the same as reversing the result)
        """

        graph = nx.Graph()
        if full:
            graph.add_nodes_from(self.graph)
        test = _false_edge if reverse else _true_edge
        graph.add_edges_from([e for e in self.graph.edges if test(self.graph, e)])
        for component in nx.connected_components(graph):
            yield next(self.all_topological_sorts(component, reverse))

    def teach(self, pair: Tuple, value: bool) -> None:
        """Teach the model that the ID with pair has a value

        :param pair:    Pair of IDs of documents
        :param value:   Value to teach
        """
        a, b = pair
        if a == b or self.graph.has_edge(a, b):
            return
        if value:
            # a < b
            lte_a = self._lte(a)
            gte_b = self._gte(b)
            for i in lte_a:
                for j in gte_b:
                    self._set_value(pair, i, j, True)
        else:
            # a >= b
            gt_a = self._gt(a)
            gte_a = self._gte(a)
            lt_b = self._lt(b)
            lte_b = self._lte(b)
            for i in gt_a:
                for j in lte_b:
                    self._set_value(pair, j, i, True)
            for i in gte_a:
                for j in lt_b:
                    self._set_value(pair, j, i, True)
            for i in gte_a:
                for j in lte_b:
                    self._set_value(pair, i, j, False)

    def _lt(self, x):
        members = set()
        if x in self.graph.nodes:
            for y in self.graph.predecessors(x):
                data = self.graph.get_edge_data(y, x)
                if data["value"]:
                    members.add(y)
        return members

    def _gt(self, x):
        members = set()
        if x in self.graph.nodes:
            for y in self.graph.successors(x):
                data = self.graph.get_edge_data(x, y)
                if data["value"]:
                    members.add(y)
        return members

    def _lte(self, x):
        members = {x}
        if x in self.graph.nodes:
            for y in self.graph.successors(x):
                data = self.graph.get_edge_data(x, y)
                if not data["value"]:
                    members.add(y)
        return members

    def _gte(self, x):
        members = {x}
        if x in self.graph.nodes:
            for y in self.graph.predecessors(x):
                data = self.graph.get_edge_data(y, x)
                if not data["value"]:
                    members.add(y)
        return members
