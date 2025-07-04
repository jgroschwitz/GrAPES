from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.util import get_node_by_name

EDGE_LABELS_AND_REIFICATIONS_HAND = {":time": ["be-temporally-at-91", "ARG2"],
                                ":location": ["be-located-at-91", "ARG2"],
                                ":quant": ["have-quant-91", "ARG2"],
                                ":degree": ["have-degree-91", "ARG2"],}

EDGE_LABELS_AND_REIFICATIONS_BERT = [[":purpose", "have-purpose-91"],
    [":beneficiary", "benefit-01"],
    [":duration", "last-01"],
    [":topic", "concern-02"],
    [":accompanier", "accompany-01"],
    [":manner", "have-manner-91"],
    [":instrument", "have-instrument-91"],
    [":location", "be-located-at-91"],
    [":time", "be-temporally-at-91"]]

IDS_FOR_EXACT_MATCH = [
    "anna_bert.tsv-0",
    "anna_bert.tsv-1",
    "anna_bert.tsv-2",
    "anna_bert.tsv-3"
]



def is_relation_present_in_graph(edge_label, graph, reification):
    return len(graph.edges(role=edge_label)) >= 1 \
           or len([inst for inst in graph.instances() if inst.target == reification]) >= 1

def get_target_instances(edge_label, graph):
    """
    Returns all nodes which are either pointed at by an edge with the given edge_label, or a reification of that edge.
    :param edge_label:
    :param graph:
    :return:
    """
    ret = []
    for edge in graph.edges(role=edge_label):
        ret.append(get_node_by_name(edge.target, graph))
    reification = EDGE_LABELS_AND_REIFICATIONS_HAND[edge_label][0]
    reification_target_edge_label = EDGE_LABELS_AND_REIFICATIONS_HAND[edge_label][1]
    for inst in graph.instances():
        if inst.target == reification:
            for edge in graph.edges(source=inst.source, role=reification_target_edge_label):
                ret.append(get_node_by_name(edge.target, graph))
    return ret


class WordDisambiguationBertsMouth(CategoryEvaluation):
    def update_results(self, gold_amr, predicted_amr, target=None, predictions_for_comparison=None):

        if gold_amr.metadata["suppl"] in IDS_FOR_EXACT_MATCH:
            if equals_modulo_isomorphy(gold_amr, predicted_amr):
                self.add_success(gold_amr, predicted_amr)
            else:
                self.add_fail(gold_amr, predicted_amr)
        else:
            found_relation = False
            for edge_label, reification in EDGE_LABELS_AND_REIFICATIONS_BERT:
                if is_relation_present_in_graph(edge_label, gold_amr, reification):
                    found_relation = True
                    if is_relation_present_in_graph(edge_label, predicted_amr, reification):
                        self.add_success(gold_amr, predicted_amr)
                    else:
                        self.add_fail(gold_amr, predicted_amr)
                    break
            if not found_relation:
                root_label = get_node_by_name(gold_amr.top, gold_amr).target
                if len([inst for inst in predicted_amr.instances() if inst.target == root_label]) >= 1:
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)


class WordDisambiguationHandcrafted(CategoryEvaluation):
    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):
        """
        This is the main function for evaluating word disambiguation for the handwritten sentences (and a few
        from the test corpus).
        """
        try:
            label = gold_amr.metadata["label"]
            if " " in label:
                # print("first case")
                edge_label, target_label = gold_amr.metadata["label"].split(" ")
                target_instances = get_target_instances(edge_label, predicted_amr)
                found = False
                for target_instance in target_instances:
                    if target_instance.target == target_label:
                        self.add_success(gold_amr, predicted_amr)
                        found = True
                        break
                if not found:
                    self.add_fail(gold_amr, predicted_amr)

            elif label.startswith(":"):
                # print("second case")
                edge_label = label
                if is_relation_present_in_graph(edge_label, predicted_amr, EDGE_LABELS_AND_REIFICATIONS_HAND[edge_label][0]):
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)
            else:
                # print("third case")
                node_label = label
                if len([inst for inst in predicted_amr.instances() if inst.target == node_label]) >= 1:
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)
        except KeyError as e:
            print("error:", e)
            print(gold_amr.metadata)
            raise e
