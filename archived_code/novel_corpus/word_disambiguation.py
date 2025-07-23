import re

from penman import Graph, load
from typing import List
from evaluation.util import get_node_by_name
from evaluation.graph_matcher import equals_modulo_isomorphy


EDGE_LABELS_AND_REIFICATIONS = {":time": ["be-temporally-at-91", "ARG2"],
                                ":location": ["be-located-at-91", "ARG2"],
                                ":quant": ["have-quant-91", "ARG2"],
                                ":degree": ["have-degree-91", "ARG2"],}



def evaluate_word_disambiguation(gold_amrs: List[Graph], predicted_amrs: List[Graph]):
    """
    This is the main function for evaluating word disambiguation for the handwritten sentences (and a few
    from the test corpus).
    """
    sample_size = 0
    success_count = 0
    for gold, pred in zip(gold_amrs, predicted_amrs):

        if re.match(pattern="word_disambiguation_[0-9]+", string=gold.metadata["id"]) is None:
            continue
        # print(gold.metadata["label"])
        sample_size += 1
        try:
            label = gold.metadata["label"]
            if " " in label:
                # print("first case")
                edge_label, target_label = gold.metadata["label"].split(" ")
                target_instances = get_target_instances(edge_label, pred)
                for target_instance in target_instances:
                    if target_instance.target == target_label:
                        success_count += 1
                        break

            elif label.startswith(":"):
                # print("second case")
                edge_label = label
                if is_relation_present_in_graph(edge_label, pred, EDGE_LABELS_AND_REIFICATIONS[edge_label][0]):
                    success_count += 1
            else:
                # print("third case")
                node_label = label
                if len([inst for inst in pred.instances() if inst.target == node_label]) >= 1:
                    success_count += 1
        except KeyError as e:
            print("error:", e)
            print(gold.metadata)
            raise e

    return success_count, sample_size


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
    reification = EDGE_LABELS_AND_REIFICATIONS[edge_label][0]
    reification_target_edge_label = EDGE_LABELS_AND_REIFICATIONS[edge_label][1]
    for inst in graph.instances():
        if inst.target == reification:
            for edge in graph.edges(source=inst.source, role=reification_target_edge_label):
                ret.append(get_node_by_name(edge.target, graph))
    return ret


def main():
    gold_amrs = load("../corpus/word_disambiguation.txt")
    predicted_amrs = load("../corpus/word_disambiguation.txt")
    print(evaluate_word_disambiguation(gold_amrs, predicted_amrs))


if __name__ == "__main__":
    main()
