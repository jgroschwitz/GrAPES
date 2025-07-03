import re
import sys
from typing import Set

from penman import Graph, encode



def num_to_score_with_preceding_0(number):
    string_result = f"{(number*100):.0f}"
    if len(string_result) == 1:
        string_result = "0" + string_result
    return string_result


def get_source(edge, graph: Graph):
    return get_node_by_name(edge.source, graph)


def get_target(edge, graph: Graph):
    return get_node_by_name(edge.target, graph)


def get_name(name_node_name, graph):
    """
    Gets the attributes of the given node in the graph, returning them joined into a string.
        Removes quotation marks.
        Intended to be used with a "name" node and its opi attributes, so the name comes out in order
    :param name_node_name:
    :param graph: penman graph
    :return: " ".join of the attributes in order
    """
    ret = []
    for attribute in sorted(graph.attributes(source=name_node_name), key=lambda x: x.role):
        ret.append(attribute.target.replace('"', ''))
    return " ".join(ret)


def get_node_by_name(node_name: str, graph: Graph):
    return [n for n in graph.instances() if n.source == node_name][0]


def get_other_node(node, edge, graph: Graph):
    if node.source == edge.source:
        return get_target(edge, graph)
    else:
        return get_source(edge, graph)


def strip_sense(node_label: str):
    # check if nodel_label matches regex ".*-[0-9][0-9]"
    if re.match(r".*-[0-9][0-9]", node_label):
        return node_label[:-3]
    else:
        return node_label


def copy_graph(graph: Graph):
    return Graph(graph.triples.copy(), top=graph.top)


def remove_edge(graph: Graph, edge):
    """
    Removes an edge from a graph. Note that this does not remove the nodes that are connected by the edge. Changes
    the graph in place, and returns nothing.
    :param graph:
    :param edge:
    :return:
    """
    graph.triples.remove(edge)


def with_edge_removed(graph: Graph, edge):
    """
    Returns a copy of the graph with the given edge removed.
    :param graph:
    :param edge:
    :return:
    """
    new_graph = copy_graph(graph)
    remove_edge(new_graph, edge)
    return new_graph


def get_connected_subgraph_from_node(node, graph: Graph):
    connected_subgraph = Graph()
    connected_subgraph.triples.append(node)
    connected_subgraph.top = node.source
    _explore_node(node, {node.source}, graph, connected_subgraph.triples.append)
    return connected_subgraph


def _explore_node(node, seen: Set[str], graph: Graph, triple_function):
    for edge in graph.edges(source=node.source):
        if edge.target not in seen:
            seen.add(edge.target)
            triple_function(edge)
            triple_function(get_target(edge, graph))
            _explore_node(get_target(edge, graph), seen, graph, triple_function)
    for edge in graph.edges(target=node.source):
        if edge.source not in seen:
            seen.add(edge.source)
            triple_function(edge)
            triple_function(get_source(edge, graph))
            _explore_node(get_source(edge, graph), seen, graph, triple_function)
    for attr in graph.attributes(source=node.source):
        triple_function(attr)


def get_raw_amr_string(graph: Graph):
    ret = ""
    for line in encode(graph).split("\n"):
        if not line.startswith("#"):
            ret += " " + line.strip()
    return ret.strip()


def is_opi_edge(edge):
    return re.match(r":op[0-9]+", edge.role)


def is_unseen_coord_opi_edge(edge):
    '''
    In the AMRBank 3.0 training set, we have seen a conjunction with 19 conjuncts (i.e. up to :op19). This checks
    if the edge is an edge with label :op20+. (Does NOT actually check if this is a coordination)
    :param edge:
    :return:
    '''
    if re.match(r":op[0-9]+", edge.role):
        number = int(edge.role[3:])
        return number >= 20
    else:
        return False


def get_node_name_for_gold_label(gold_label, gold_amr, is_attribute):
    if is_attribute:
        for attr in gold_amr.attributes():
            if attr.target == gold_label:
                return attr.source
        for label_alternatives in gold_label.split(" "):
            for attr in gold_amr.attributes():
                if attr.target == label_alternatives:
                    return attr.source
    else:
        for instance in gold_amr.instances():
            if instance.target == gold_label:
                return instance.source
        for label_alternatives in gold_label.split(" "):
            for instance in gold_amr.instances():
                if instance.target == label_alternatives:
                    return instance.source
    return None


def filter_amrs_for_name(name, gold_graphs, predicted_graphs, fail_ok=False):
    regular_expression = name+"_[0-9]+"
    gold_graphs_filtered = []
    predicted_graphs_filtered = []
    for g, p in zip(gold_graphs, predicted_graphs):
        if re.match(regular_expression, g.metadata["id"]):
            gold_graphs_filtered.append(g)
            predicted_graphs_filtered.append(p)
    if len(gold_graphs_filtered) == 0 and fail_ok:
        print("WARNING: didn't find any AMRs for using filter_amrs_for_name", name, file=sys.stderr)
    else:
        assert len(gold_graphs_filtered) > 0, f"Corpus does not contain any AMRs with ID matching {regular_expression}"

    return gold_graphs_filtered, predicted_graphs_filtered


SANITY_CHECK = "Sanity check"


def num_to_score(number):
    return f"{(number * 100):.0f}"
