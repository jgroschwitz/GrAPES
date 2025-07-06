from collections import deque
from typing import Dict

import penman
from penman import Graph

from evaluation.file_utils import get_graph_for_node_string
from evaluation.util import get_node_by_name, get_target, get_source, strip_sense, get_other_node


def count_edges_between_two_nodes_unlabeled(graph, node1, node2):
    return len(graph.edges(source=node1, target=node2)) + len(graph.edges(source=node2, target=node1))


class SubgraphMatcher:

    def __init__(self, full_graph, subgraph, match_edge_labels: bool = True, match_senses: bool = True):
        """
        Checks whether the given subgraph is contained in the given full graph.
        :param full_graph: The full graph, which may contain the subgraph.
        :param subgraph: The potential subgraph, whose sub-graph-ness we're testing.
        :param match_edge_labels: Whether edge labels should be taken into account.
        :param match_senses: Whether senses should be taken into account.
        """
        self.full_graph = full_graph
        self.subgraph = subgraph
        self.match_edge_labels = match_edge_labels
        self.match_senses = match_senses

    def contains_modulo_isomorphy(self) -> bool:
        mappings = []
        node_agenda = deque()
        node_agenda.append(self.subgraph.top)
        seen_nodenames_in_subgraph = set()
        seen_nodenames_in_subgraph.add(self.subgraph.top)
        # initialize mappings with possible mappings of top node
        for node in self.full_graph.instances():
            if self.do_nodes_match(node, get_node_by_name(self.subgraph.top, self.subgraph), dict()):
                mappings.append({self.subgraph.top: node.source})
        if len(mappings) == 0:
            return False
        while len(node_agenda) > 0:
            node_name = node_agenda.popleft()
            for edge in self.subgraph.edges(source=node_name):
                if edge.target not in seen_nodenames_in_subgraph:
                    seen_nodenames_in_subgraph.add(edge.target)
                    node_agenda.append(edge.target)
                    new_mappings = []
                    for mapping in mappings:
                        new_mappings.extend(self.extend_mapping(mapping, edge, False))
                    if len(new_mappings) == 0:
                        return False
                    mappings = new_mappings
            for edge in self.subgraph.edges(target=node_name):
                if edge.source not in seen_nodenames_in_subgraph:
                    seen_nodenames_in_subgraph.add(edge.source)
                    node_agenda.append(edge.source)
                    new_mappings = []
                    for mapping in mappings:
                        new_mappings.extend(self.extend_mapping(mapping, edge, True))
                    if len(new_mappings) == 0:
                        return False
                    mappings = new_mappings

        # need to have at least one valid mapping now
        for mapping in mappings:
            if self.is_mapping_valid(mapping):
                return True
        return False

    def is_mapping_valid(self, mapping: Dict) -> bool:
        for edge in self.subgraph.edges():
            if count_edges_between_two_nodes_unlabeled(self.subgraph, edge.source, edge.target) \
                    > count_edges_between_two_nodes_unlabeled(self.full_graph,
                                                              mapping[edge.source],
                                                              mapping[edge.target]):
                return False
        return True

    def extend_mapping(self, node_mapping: Dict, edge_in_subgraph, invert_edge: bool):
        if invert_edge:
            old_node = get_target(edge_in_subgraph, self.subgraph)
        else:
            old_node = get_source(edge_in_subgraph, self.subgraph)
        assert old_node.source in node_mapping
        if invert_edge:
            new_node = get_source(edge_in_subgraph, self.subgraph)
        else:
            new_node = get_target(edge_in_subgraph, self.subgraph)
        old_node_full_graph = get_node_by_name(node_mapping[old_node.source], self.full_graph)
        candidate_edges = self.get_candidate_extension_edges(edge_in_subgraph, old_node_full_graph, invert_edge)
        new_mappings = []
        for candidate_edge in candidate_edges:
            candidate_node_full_graph = get_other_node(old_node_full_graph, candidate_edge, self.full_graph)
            if self.do_nodes_match(candidate_node_full_graph, new_node, node_mapping):
                new_mapping = node_mapping.copy()
                new_mapping[new_node.source] = candidate_node_full_graph.source
                new_mappings.append(new_mapping)
        return new_mappings

    def get_candidate_extension_edges(self, edge_in_subgraph, old_node_full_graph, invert_edge):
        if self.match_edge_labels:
            if invert_edge:
                candidate_edges = self.full_graph.edges(target=old_node_full_graph.source, role=edge_in_subgraph.role)
            else:
                candidate_edges = self.full_graph.edges(source=old_node_full_graph.source, role=edge_in_subgraph.role)
        else:
            # just take all adjacent edges
            candidate_edges = self.full_graph.edges(target=old_node_full_graph.source) \
                              + self.full_graph.edges(source=old_node_full_graph.source)
        return candidate_edges

    def do_attributes_match(self, node_in_full_graph, node_in_subgraph):
        # only check attributes of subgraph node. Not all attributes in the full graph need to be in the subgraph.
        for attr2 in self.subgraph.attributes(source=node_in_subgraph.source):
            if not any(self.does_attribute_match(attr1, attr2)
                       for attr1 in self.full_graph.attributes(source=node_in_full_graph.source)):
                return False
        return True

    def does_attribute_match(self, attr1, attr2):
        return self.do_edge_labels_match(attr1, attr2)\
               and attr1.target.replace("\"", "") == attr2.target.replace("\"", "")

    def do_edge_labels_match(self, edge_in_full_graph, edge_in_subgraph):
        if self.match_edge_labels:
            return edge_in_full_graph.role == edge_in_subgraph.role
        else:
            return True

    def do_nodes_match(self, node_in_full_graph, node_in_subgraph, mapping: Dict[str, str]):

        if self.match_senses:
            do_labels_match = node_in_full_graph.target == node_in_subgraph.target
        else:
            do_labels_match = strip_sense(node_in_full_graph.target) == strip_sense(node_in_subgraph.target)
        return do_labels_match \
            and self.do_attributes_match(node_in_full_graph, node_in_subgraph) \
            and self.do_reentrant_edges_match(node_in_full_graph, node_in_subgraph, mapping)

    def do_reentrant_edges_match(self, node_in_full_graph, node_in_subgraph, mapping: Dict[str, str]):
        for edge_in_subgraph in self.subgraph.edges(source=node_in_subgraph.source):
            if edge_in_subgraph.target in mapping:
                if self.match_edge_labels:
                    matching_edges_in_full_graph = self.full_graph.edges(source=node_in_full_graph.source,
                                                                         role=edge_in_subgraph.role,
                                                                         target=mapping[edge_in_subgraph.target])
                else:
                    matching_edges_in_full_graph = self.full_graph.edges(source=node_in_full_graph.source,
                                                                         target=mapping[edge_in_subgraph.target]) + \
                                                    self.full_graph.edges(target=node_in_full_graph.source,
                                                                          source=mapping[edge_in_subgraph.target])
                if len(matching_edges_in_full_graph) == 0:
                    # having more than 1 matchin edge is fine, since we are only checking for subgraphity
                    return False
        for edge_in_subgraph in self.subgraph.edges(target=node_in_subgraph.source):
            if edge_in_subgraph.source in mapping:
                if self.match_edge_labels:
                    matching_edges_in_full_graph = self.full_graph.edges(target=node_in_full_graph.source,
                                                                         role=edge_in_subgraph.role,
                                                                         source=mapping[edge_in_subgraph.source])
                else:
                    # ignore both edge labels and edge direction
                    # Note: if the subgraph has multiple edges between two nodes, and the full graph has fewer,
                    #  this will give a false positive. We are checking for that in postprocessing
                    #  (see self.is_mapping_valid). This is a bit inefficient since we keep incorrect mappings around,
                    #  but the case should be so rare that it doesn't really matter.
                    matching_edges_in_full_graph = self.full_graph.edges(target=node_in_full_graph.source,
                                                                         source=mapping[edge_in_subgraph.source]) + \
                                                    self.full_graph.edges(source=node_in_full_graph.source,
                                                                          target=mapping[edge_in_subgraph.source])

                if len(matching_edges_in_full_graph) == 0:
                    # having more than 1 matchin edge is fine, since we are only checking for subgraphity
                    return False
        # don't need to check that all edges in full graph match, since we are only checking for subgraphity
        return True


def contains_subgraph_modulo_isomorphy(graph: Graph, subgraph: Graph,
                                       match_edge_labels: bool = True, match_senses: bool = True) -> bool:
    """
    Takes a graph and a potential subgraph, and returns true if the full graph contains the subgraph (modulo isomorphy, root).
    :param graph:
    :param subgraph:
    :param match_edge_labels: if false, edge labels and edge directions are ignored
    :param match_senses: if false, PropBank senses are ignored
    :return:
    """
    matcher = SubgraphMatcher(graph, subgraph, match_edge_labels=match_edge_labels, match_senses=match_senses)
    return matcher.contains_modulo_isomorphy()


def equals_modulo_isomorphy(graph1: Graph, graph2: Graph, match_edge_labels: bool = True, match_senses: bool = True):
    ret = contains_subgraph_modulo_isomorphy(graph1, graph2,
                                             match_edge_labels=match_edge_labels, match_senses=match_senses)\
          and contains_subgraph_modulo_isomorphy(graph2, graph1,
                                                 match_edge_labels=match_edge_labels, match_senses=match_senses)
    # if not ret:
    #     print("graph1", encode(graph1))
    #     print("graph2", encode(graph2))
    #     print()
    return ret


def check_fragment_existence(graph_fragment_string: str, predicted_amr,
                             match_edge_labels: bool = True, match_senses: bool = True) -> bool:
    """
    Checks if the given graph fragment exists in the predicted AMR.
    :param match_senses:
    :param match_edge_labels:
    :param graph_fragment_string: Graph fragment in string form, in the "node_string" format we use in tsv files.
    See the documentation for get_graph_for_node_string for details on that format.
    :param predicted_amr:
    :return:
    """
    graph_fragment = get_graph_for_node_string(graph_fragment_string)
    ret = contains_subgraph_modulo_isomorphy(predicted_amr, graph_fragment, match_edge_labels, match_senses)
    # if not ret:
    #     print("graph_fragment_string", graph_fragment_string)
    #     print("graph_fragment", penman.encode(graph_fragment))
    #     print("predicted_amr", penman.encode(predicted_amr))
    #     print()
    return ret


def check_edge_label_existence(edge_label: str, predicted_amr: Graph) -> bool:
    return len(predicted_amr.edges(role=edge_label)) > 0


def main():
    """
    Essentially test cases.
    :return:
    """
    # predicted_graph = penman.decode("(z0 / and :op1 (z1 / ask-02 :ARG0 (z2 / girl) :ARG1 (z3 / force-01 :ARG0 (z6 / politician) "
    #               ":ARG1 (z4 / doctor) :ARG2 (z5 / jump-03 :ARG0 z4)) :ARG2 z6) :op2 (z7 / eat-01 :ARG0 z2))")
    # gold_graph = penman.decode("(u_531 / and  :op1 (u_535 / ask-02  :ARG0 (r / girl  :ARG0-of (u_537 / eat-01  :op2-of u_531)) " \
    #              " :ARG2 (u_538 / politician  :ARG0-of (u_532 / force-01  :ARG1 (u_536 / doctor  :ARG0-of" \
    #              " (u_534 / jump-01  :ARG2-of u_532))  :ARG1-of u_535))))")
    #
    # print("graphs matching fully (identical graphs)")
    # print(equals_modulo_isomorphy(gold_graph, gold_graph, match_edge_labels=True, match_senses=True))
    # print(equals_modulo_isomorphy(gold_graph, gold_graph, match_edge_labels=False, match_senses=True))
    # print(equals_modulo_isomorphy(gold_graph, gold_graph, match_edge_labels=True, match_senses=False))
    # print(equals_modulo_isomorphy(gold_graph, gold_graph, match_edge_labels=False, match_senses=False))
    #
    #
    # print("graphs matching edge labels but not senses")
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=False))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=False))
    #
    #
    # predicted_graph = penman.decode("(z0 / and :op3 (z1 / ask-02 :ARG0 (z2 / girl) :ARG1 (z3 / force-01 :ARG0 (z6 / politician) "
    #               ":ARG1 (z4 / doctor :value 4) :ARG2 (z5 / jump-03 :ARG0 z4)) :ARG2 z6) :op2 (z7 / eat-01 :ARG0 z2))")
    # gold_graph = penman.decode("(u_531 / and  :op1 (u_535 / ask-02  :ARG0 (r / girl  :ARG0-of (u_537 / eat-01  :op2-of u_531)) " \
    #              " :ARG2 (u_538 / politician  :ARG0-of (u_532 / force-01  :ARG1 (u_536 / doctor :quant 4  :ARG0-of" \
    #              " (u_534 / jump-01  :ARG2-of u_532))  :ARG1-of u_535))))")
    #
    # print("graphs matching neither edge labels nor senses")
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=False))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=False))
    #
    # predicted_graph = penman.decode("(u_531 / and  :op1 (u_535 / ask-02  :ARG0 (r / girl  :ARG1-of (u_537 / eat-01  :op2-of u_531)) " \
    #              " :ARG2 (u_538 / politician  :ARG0 (u_532 / force-01  :ARG1 (u_536 / doctor :quant 4  :ARG0-of" \
    #              " (u_534 / jump-01  :ARG2-of u_532))  :ARG1-of u_535))))")
    #
    # print("graphs matching senses but not edge labels")
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=False))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=False))
    #
    #
    # predicted_graph = penman.decode("(u_531 / and  :op1 (u_535 / ask-02  :ARG0-of (r / girl  :ARG0-of (u_537 / eat-01  :op2-of u_531)) " \
    #              " :ARG2 (u_538 / politician  :ARG0 (u_532 / force-01  :ARG1 (u_536 / doctor :quant 4  :ARG0-of" \
    #              " (u_534 / jump-01  :ARG2-of u_532))  :ARG1 u_535))))")
    #
    # print("graphs matching senses and edge labels, but not edge-directions")
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=False))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=False))
    #
    # print("identical graphs but additional reentrant edge")
    #
    # gold_graph = penman.decode("(a / a :ARG0 (b / b))")
    # predicted_graph = penman.decode("(a / a :ARG0 (b / b) :ARG1 b)")
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=True))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=True, match_senses=False))
    # print(equals_modulo_isomorphy(predicted_graph, gold_graph, match_edge_labels=False, match_senses=False))
    #
    # full_graph = penman.decode("(n / need-01 :ARG0 (w / we) :ARG1 (h / help-01) :concession" +
    #                            " (g / get-02 :polarity -:ARG0 w :ARG1 h :mod (e / ever)))")
    # subgraph = penman.decode("(g / get-02)")
    # assert contains_subgraph_modulo_isomorphy(full_graph, subgraph, match_edge_labels=True, match_senses=True)


    full_graph = penman.decode("""(a3 / and
    :op1 (d2 / defeat-01
             :ARG0 (p7 / person
                       :wiki -
                       :name (n / name
                                :op1 "XunXuan"
                                :op2 "Cao")
                       :mod (l / level
                               :ord (o / ordinal-entity
                                       :value 9))
                       :part-of (t / team
                                   :mod (c6 / country
                                            :wiki "Korea"
                                            :name (n4 / name
                                                      :op1 "Korea"))))
             :ARG1 (p10 / person
                        :wiki -
                        :name (n5 / name
                                  :op1 "Jing"
                                  :op2 "Lui")
                        :mod (l4 / level
                                 :ord (o4 / ordinal-entity
                                          :value 5))
                        :part-of (t2 / team
                                     :mod (c7 / country
                                              :wiki "China"
                                              :name (n8 / name
                                                        :op1 "China")))))
    :op2 (d3 / defeat-01
             :ARG0 (p8 / person
                       :wiki -
                       :name (n2 / name
                                 :op1 "Changhao"
                                 :op2 "Li")
                       :mod (l2 / level
                                :ord (o2 / ordinal-entity
                                         :value 7))
                       :part-of t)
             :ARG1 (p11 / person
                        :wiki "Ma_Xiaochun"
                        :name (n6 / name
                                  :op1 "Xiaocun"
                                  :op2 "Ma")
                        :mod l
                        :part-of t2))
    :op3 (d / defeat-01
            :ARG0 (p9 / person
                      :wiki -
                      :name (n3 / name
                                :op1 "Changhe"
                                :op2 "Liu")
                      :mod (l3 / level
                               :ord (o3 / ordinal-entity
                                        :value 6))
                      :part-of t)
            :ARG1 (p12 / person
                       :wiki -
                       :name (n7 / name
                                 :op1 "Jianhong"
                                 :op2 "Wang")
                       :mod l2
                       :part-of t2)))""")
    subgraph = penman.decode("(l / level :ord (o / ordinal-entity :value 7)))")
    print(contains_subgraph_modulo_isomorphy(full_graph, subgraph, match_edge_labels=True, match_senses=True))

if __name__ == "__main__":
    main()
