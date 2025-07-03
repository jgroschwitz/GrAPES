from typing import List, Tuple, Set

import penman

from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, PREREQS, UNLABELLED
from evaluation.util import strip_sense, get_source, get_target, filter_amrs_for_name

import re

from penman import Graph

# Relation	Reification	Domain	Range	Example
# :accompanier	accompany-01	:ARG0	:ARG1	“she's with him”
# :age	age-01	:ARG1	:ARG2	“she's 41 years old”
# :beneficiary	benefit-01	:ARG0	:ARG1	“the 5k run is for kids”
# :concession	have-concession-91	:ARG1	:ARG2	“he came despite of her”
# :condition	have-condition-91	:ARG1	:ARG2	“he comes if she comes”
# :degree	have-degree-92	:ARG1	:ARG2	“very tall” (intensifier or downtoner)
# :destination	be-destined-for-91	:ARG1	:ARG2	“i'm off to Atlanta”
# :duration	last-01	:ARG1	:ARG2	“it's 15 minutes long”
# :example	exemplify-01	:ARG0	:ARG1	“cities such as Atlanta”
# :extent	have-extent-91	:ARG1	:ARG2	“trip was 2500 miles”
# :frequency	have-frequency-91	:ARG1	:ARG2	“he came three times”
# :instrument	have-instrument-91	:ARG1	:ARG2	“forks are for eating”
# :li	have-li-91	:ARG1	:ARG2	“(B)”
# :location	be-located-at-91	:ARG1	:ARG2	“she's not here”
# :manner	have-manner-91	:ARG1	:ARG2	“it was done quickly”
# :mod	have-mod-91	:ARG1	:ARG2	“he is half Chinese"
# :name	have-name-91	:ARG1	:ARG2	“the city formerly named Constantinople”
# :ord	have-ord-91	:ARG1	:ARG2	“I don't know whether it was his first loss.”
# :part	have-part-91	:ARG1	:ARG2	“the roof of the house”
# :polarity	have-polarity-91	:ARG1	:ARG2	“I don't know.”
# :poss	own-01, have-03	:ARG1	:ARG0	“that dog's not mine”
# :purpose	have-purpose-91	:ARG1	:ARG2	“it's to eliminate bugs”
# :quant	have-quant-91	:ARG1	:ARG2	“there are 4 rabbits”
# :source	be-from-91	:ARG1	:ARG2	“she's from Ipanema”
# :subevent	have-subevent-91	:ARG1	:ARG2	“presentation at a conference”
# :time	be-temporally-at-91	:ARG1	:ARG2	“the party is on friday”
# :topic	concern-02	:ARG0	:ARG1	“the show's about me”
# :value	have-value-91	:ARG1	:ARG2	“The phone number is 1-800-555-1223.”

REIFICATION_QUADRUPLES = [
    (":accompanier", "accompany-01", ":ARG0", ":ARG1"),
    (":age", "age-01", ":ARG1", ":ARG2"),
    (":beneficiary", "benefit-01", ":ARG0", ":ARG1"),
    (":concession", "have-concession-91", ":ARG1", ":ARG2"),
    (":poss", "own-01,have-03", ":ARG1", ":ARG0"),
    (":instrument", "have-instrument-91", ":ARG1", ":ARG2"),
    (":time", "be-temporally-at-91", ":ARG1", ":ARG2"),
    (":location", "be-located-at-91", ":ARG1", ":ARG2"),
    (":source", "be-from-91", ":ARG1", ":ARG2"),
    (":destination", "be-destined-for-91", ":ARG1", ":ARG2"),
    (":purpose", "have-purpose-91", ":ARG1", ":ARG2"),
    (":example", "exemplify-01", ":ARG0", ":ARG1"),
    (":extent", "have-extent-91", ":ARG1", ":ARG2"),
    (":manner", "have-manner-91", ":ARG1", ":ARG2"),
    (":cause", "cause-01", ":ARG0", ":ARG1"),

]

edge2reification_map = {t[0]: (t[1], t[2], t[3]) for t in REIFICATION_QUADRUPLES}
node2edge_map = {t[1]: (t[0]) for t in REIFICATION_QUADRUPLES}
node2reification_edges_map = {t[1]: (t[2], t[3]) for t in REIFICATION_QUADRUPLES}

# can add nodes here, only to node2reification_edges_map
node2reification_edges_map["author-01"] = (":ARG0", ":ARG1")


class PPAttachment(CategoryEvaluation):
    """
    We look for edges or reification-style common parents
    The "source" throughout is the root of the modified subgraph, and the "target" the root of the modifier,
        except with :poss, where the possessor is the source and the target is the possessed thing,
        and author-01 because Jonas wrote it in backwards to confuse me. Since the nodes and edges in
        question are extracted from the gold graph, this works with no special cases
    Attributes:
        prerequisites_counter: the number of graphs with nodes labelled with the labels of interest, modulo sources
        unlabeled_counter: the number of graphs with any edge from the desired source to target, modulo senses
        labeled_counter: the number of graphs with a correctly-labelled edge or reification node
                            linking the desired source to target. Includes senses
        total_counter: the number of graphs checked
        gold_directory: list of gold graphs
        predictions_directory: list of predicted graphs

    """
    def __init__(self, gold_amrs, predicted_amrs, root_dir, info, predictions_directory=None, do_error_analysis=False):
        """
        Pragmatic attachments of ambiguous PPs
        PP Attachments come from multiple files, so if they're not already in the given graphs, we try to get them.
        """
        super().__init__(gold_amrs, predicted_amrs, root_dir, info, predictions_directory, do_error_analysis)
        # if we read in the unused PP directory instead of the whole full_corpus, replace it with the real ones
        # These have ids pp_attachment_n
        if self.gold_amrs[0].metadata['id'].startswith(self.category_metadata.subcorpus_filename):
            print("Reading in additional files")
            self.gold_amrs, self.predicted_amrs = self.get_additional_graphs(read_in=True)
        if len(self.gold_amrs) != len(self.predicted_amrs) or len(self.gold_amrs) ==0:
            raise Exception("Different number of AMRs or 0")

    def measure_unlabelled_edges(self):
        return True

    # def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph]):
    #     self.prerequisites_counter = 0
    #     self.unlabeled_counter = 0
    #     self.labeled_counter = 0
    #     self.total_counter = 0
    #     self.gold_amrs = gold_amrs
    #     self.predicted_amrs = predicted_amrs

    # def evaluate_all(self):
    #     self._run_all_evaluations_and_update_internal_counters()
    #     assert self.total_counter > 0, "No graphs found"
    #     return self.prerequisites_counter / self.total_counter, self.unlabeled_counter / self.total_counter, \
    #            self.labeled_counter / self.total_counter

    def _get_all_results(self):
        g, p = self.filter_graphs()
        print(len(g))
        self.evaluate_see_with_graphs()
        self.evaluate_read_by_graphs()
        self.evaluate_bought_for_graphs()
        self.evaluate_keep_from_graphs()
        self.evaluate_give_up_in_graphs()
        print(len(self.gold_amrs))

    def evaluate_see_with_graphs(self):
        golds, predictions = filter_amrs_for_name("see_with", self.gold_amrs, self.predicted_amrs, fail_ok=True)
        assert len(golds) == len(predictions), f"{len(golds)} gold graphs found"
        self.evaluate_graphs(predictions, golds, edge_labels_to_evaluate={":poss", ":instrument"})

    def evaluate_read_by_graphs(self):
        golds, predictions = filter_amrs_for_name("read_by", self.gold_amrs, self.predicted_amrs, fail_ok=True)
        assert len(golds) == len(predictions), f"{len(golds)} gold graphs found"
        self.evaluate_graphs(predictions, golds, edge_labels_to_evaluate={":time", ":manner"},
                             node_labels_to_evaluate={"author-01"})

    def evaluate_bought_for_graphs(self):
        golds, predictions = filter_amrs_for_name("bought_for", self.gold_amrs, self.predicted_amrs, fail_ok=True)
        assert len(golds) == len(predictions), f"{len(golds)} gold graphs found"
        self.evaluate_graphs(predictions, golds, edge_labels_to_evaluate={":purpose", ":ARG3"})

    def evaluate_keep_from_graphs(self):
        golds, predictions = filter_amrs_for_name("keep_from", self.gold_amrs, self.predicted_amrs, fail_ok=True)
        assert len(golds) == len(predictions), f"{len(golds)} gold graphs found"
        self.evaluate_graphs(predictions, golds, edge_labels_to_evaluate={":source", ":ARG2"})

    def evaluate_give_up_in_graphs(self):
        golds, predictions = filter_amrs_for_name("give_up_in", self.gold_amrs, self.predicted_amrs, fail_ok=True)
        assert len(golds) == len(predictions), f"{len(golds)} gold graphs found"
        self.evaluate_graphs(predictions, golds, edge_labels_to_evaluate={":time", ":topic"},
                             node_labels_to_evaluate={"cause-01"})

    def evaluate_graphs(self, predictions: List[Graph], golds: List[Graph], edge_labels_to_evaluate: Set[str] = None,
                        node_labels_to_evaluate=None):
        if edge_labels_to_evaluate is None:
            edge_labels_to_evaluate = {}
        if node_labels_to_evaluate is None:
            node_labels_to_evaluate = {}
        assert len(predictions) == len(golds)
        for pred, gold in zip(predictions, golds):
            edges_in_gold = [e for edge_label in edge_labels_to_evaluate for e in gold.edges(role=edge_label)]
            nodes_in_gold = [n for n in gold.instances() if n[2] in node_labels_to_evaluate]

            # if the gold graph only contains one of the edges or nodes we want, evaluate that
            if len(edges_in_gold) + len(nodes_in_gold) == 1:
                if len(edges_in_gold) == 1:
                    self.evaluate_edge_presence(gold, edges_in_gold[0], pred)
                else:  # that is, if len(nodes_in_gold) == 1
                    self.evaluate_node_presence(gold, nodes_in_gold[0], pred)
            # otherwise, we expect that the node is a cause-01 node and the edge is a :time edge
            # in this case, :time is not the relevant edge
            else:
                if len(nodes_in_gold) == 1 and nodes_in_gold[0].target == "cause-01" and\
                            len(edges_in_gold) == 1 and edges_in_gold[0].role == ":time":
                    # print("PP attachment workaround for 'gave up in a moment of clarity'; evaluating the node."
                    #       " This is as intended.")
                    self.evaluate_node_presence(gold, nodes_in_gold[0], pred)
                else:
                    print("WARNING: No edge or node we are looking to evaluate in gold graph (or more than one)."
                          " Skipping.")
                    print(penman.encode(gold))
                    print("edges_in_gold", edges_in_gold)
                    print("nodes_in_gold", nodes_in_gold)

    def evaluate_node_presence(self, gold, gold_node, predicted_graph):
        """
        Used when the gold graph's point of interest is a node, rather than an edge.
        But most of these nodes are reifications of edges, so we also look for those if the node doesn't pan out
        :param gold: gold penman graph
        :param gold_node: node name of the node of interest in the gold graph
        :param predicted_graph: penman graph
        """
        edge_label_to_source, edge_label_to_target = node2reification_edges_map[gold_node.target]
        gold_source = get_target(gold.edges(source=gold_node.source, role=edge_label_to_source)[0], gold)
        gold_target = get_target(gold.edges(source=gold_node.source, role=edge_label_to_target)[0], gold)

        # prereqs are the source and target nodes, minus their senses
        prerequisite_source = any(strip_sense(n.target) == strip_sense(gold_source.target) for n in predicted_graph.instances())
        prerequisite_target = any(strip_sense(n.target) == strip_sense(gold_target.target) for n in predicted_graph.instances())
        prerequisites = prerequisite_source and prerequisite_target
        if prerequisites:
            self.add_success(gold, predicted_graph, PREREQS)

            # we try both reification and edge versions
            # get all common ARGi parents of "source" and "target" in the predicted graph
            # e.g. common parents of person and book
            # ignores senses
            unlabeled_node_matches = get_reification_like_nodes(predicted_graph, gold_source, gold_target)
            unlabeled_node_match = len(unlabeled_node_matches) > 0
            if unlabeled_node_match:
                self.add_success(gold, predicted_graph, UNLABELLED)
                # for some reason we look again (discarding unlabeled_node_matches), this time for one with
                # the right edge labels and the right common parent node label, including senses
                # e.g. True if these are in the predicted graph:
                # author-01 --ARG0--> book
                # author-01 --ARG1--> person
                labeled = exists_labeled_node_connection(gold_node.target,
                                                         edge_label_to_source,
                                                         edge_label_to_target,
                                                         gold_source, gold_target, predicted_graph)
                if labeled:
                    self.add_success(gold, predicted_graph)
                else:
                    self.add_fail(gold, predicted_graph)
            else:
                # try the edge label variant instead (instead of reification)
                unlabeled = exists_unlabeled_edge_match(gold_source, gold_target, predicted_graph)
                if unlabeled:
                    self.add_success(gold, predicted_graph, UNLABELLED)
                    # find the edge variant of this reified node, if any
                    if gold_node.target in node2edge_map:
                        edge_label = node2edge_map[gold_node.target]
                        # Includes senses
                        labeled = exists_labeled_edge_match(edge_label, gold_source, gold_target, predicted_graph)
                        if labeled:
                            self.add_success(gold, predicted_graph)
                        else:
                            self.add_fail(gold, predicted_graph)
                        handled = True
                    else:
                        self.add_fail(gold, predicted_graph)
                else:
                    self.add_fail(gold, predicted_graph)
                    self.add_fail(gold, predicted_graph, UNLABELLED)
        else:
            self.add_fail(gold, predicted_graph)
            self.add_fail(gold, predicted_graph, UNLABELLED)
            self.add_fail(gold, predicted_graph, PREREQS)

        # else:
        #     print(f"PP attachment prerequisites missed: {gold_source.target}, {gold_target.target}")

    def evaluate_edge_presence(self, gold, gold_edge, predicted_graph):
        gold_source = get_source(gold_edge, gold)
        gold_target = get_target(gold_edge, gold)
        # print("Gold poss edge: " + str(gold_edge))
        prerequisite_source = any(strip_sense(n.target) == strip_sense(gold_source.target) for n in predicted_graph.instances())
        prerequisite_target = any(strip_sense(n.target) == strip_sense(gold_target.target) for n in predicted_graph.instances())
        prerequisites = prerequisite_source and prerequisite_target
        if prerequisites:
            self.add_success(gold, predicted_graph, PREREQS)

            unlabeled = exists_unlabeled_edge_match(gold_source, gold_target, predicted_graph)
            reification_nodes = get_reification_like_nodes(predicted_graph, gold_source, gold_target)
            unlabeled_reified = len(reification_nodes) > 0
            if unlabeled or unlabeled_reified:
                self.add_success(gold, predicted_graph, UNLABELLED)
                labeled = None
                if unlabeled:
                    labeled = exists_labeled_edge_match(gold_edge.role, gold_source, gold_target, predicted_graph)
                elif unlabeled_reified:
                    labeled = exists_labeled_reification_match(gold_edge, gold_source, gold_target, predicted_graph)
                if labeled:
                    self.add_success(gold, predicted_graph)
                else:
                    self.add_fail(gold, predicted_graph)
            else:
                self.add_fail(gold, predicted_graph)
                self.add_fail(gold, predicted_graph, UNLABELLED)
        else:
            self.add_fail(gold, predicted_graph)
            self.add_fail(gold, predicted_graph, UNLABELLED)
            self.add_fail(gold, predicted_graph, PREREQS)
        # else:
        #     print(f"PP attachment prerequisites missed: {gold_source.target}, {gold_target.target}")
            # else:
            #     print("\nUnlabeled error found:")
            #     print(encode(pred))
        # else:
        #     print(f"gold_source: {gold_source.target} ({prerequisite_source})")
        #     print(f"gold_target: {gold_target.target} ({prerequisite_target})")
        #     print(" ".join([n.target for n in pred.instances()]))


def exists_labeled_edge_match(gold_edge_label, gold_source, gold_target, predicted_graph):
    """
    Looks for an edge in predicted graph with:
        label gold_edge_label
        source node label gold_source
        target node label gold_target
    :return: True if any such edge exists. Ignores senses.
    """
    return any(strip_sense(get_source(e, predicted_graph).target) == strip_sense(gold_source.target) and
               strip_sense(get_target(e, predicted_graph).target) == strip_sense(gold_target.target) and e.role == gold_edge_label
               for e in predicted_graph.edges())


def exists_labeled_reification_match(gold_edge, gold_source, gold_target, predicted_graph):
    if gold_edge.role not in edge2reification_map:
        return False
    reif_label_raw = edge2reification_map[gold_edge.role][0]
    if "," in reif_label_raw:
        reif_labels = reif_label_raw.split(",")
    else:
        reif_labels = [reif_label_raw]
    reif_source_edge_label = node2reification_edges_map[reif_label_raw][0]
    reif_target_edge_label = node2reification_edges_map[reif_label_raw][1]
    return any(exists_labeled_node_connection(rl, reif_source_edge_label, reif_target_edge_label, gold_source,
                                              gold_target, predicted_graph)
               for rl in reif_labels)


def exists_labeled_node_connection(node_label, source_edge_label, target_edge_label, gold_source, gold_target,
                                   predicted_graph):
    """
    Look through the predicted graph's nodes for node n with:
        label node_label
        edge labelled source_edge_label from n to a node with label gold_source
        edge labelled target_edge_label from n to a node with label gold_target
    :return: True if any such node exists. Ignores senses.
    """
    return any(n.target == node_label and
               node_has_labeled_argument(n, source_edge_label, gold_source.target, predicted_graph) and
               node_has_labeled_argument(n, target_edge_label, gold_target.target, predicted_graph)
               for n in predicted_graph.instances())


def node_has_labeled_argument(node, edge_label, target_label, graph):
    """
    True if there is an edge in the graph node --edge_label--> target_label
    Ignores senses.
    """
    ret = any(e.role == edge_label and strip_sense(get_target(e, graph).target) == strip_sense(target_label) for e in graph.edges(source=node.source))
    return ret


def get_reification_like_nodes(graph, gold_source, gold_target):
    """
    This is reification-like because we're calling them source and target, but actually there's a common parent
    Return all common ARGi parents of the two nodes. Ignores senses.
    """
    reification_nodes = []
    # n -ARGi-> source and n -ARGi-> target
    for n in graph.instances():
        if exists_edge_with_source_role_and_target_label(graph, n.source, r":ARG[0-9]", gold_source.target) \
                and exists_edge_with_source_role_and_target_label(graph, n.source, r":ARG[0-9]", gold_target.target):
            reification_nodes.append(n)
    return reification_nodes


def exists_edge_with_source_role_and_target_label(graph, source_name, role_regex, target_label):
    """
    Ignores senses. True if there is an edge matching the pattern between source and a node with target label
    """
    return any(
        strip_sense(get_target(e, graph).target) == strip_sense(target_label)
        and re.match(role_regex, e.role) is not None
        for e in graph.edges(source=source_name))


def exists_unlabeled_edge_match(gold_source, gold_target, predicted_graph):
    """
    Ignores senses. True if there is an edge between nodes with those labels
    """
    return any(
        strip_sense(get_source(e, predicted_graph).target) == strip_sense(gold_source.target)
        and strip_sense(get_target(e, predicted_graph).target) == strip_sense(gold_target.target) for e
        in predicted_graph.edges()) \
           or any(
        strip_sense(get_source(e, predicted_graph).target) == strip_sense(gold_target.target)
        and strip_sense(get_target(e, predicted_graph).target) == strip_sense(gold_source.target) for e
        in predicted_graph.edges())


def evaluate_pp_attachments(gold_directory, predictions_directory):
    evaluator = PPAttachmentEvaluator(gold_directory, predictions_directory)
    return evaluator.evaluate_all()


def get_pp_attachment_success_counters(gold_amrs, predicted_amrs):
    evaluator = PPAttachmentEvaluator(gold_amrs, predicted_amrs)
    evaluator.evaluate_all()
    return evaluator.prerequisites_counter, evaluator.unlabeled_counter, evaluator.labeled_counter,\
           evaluator.total_counter
