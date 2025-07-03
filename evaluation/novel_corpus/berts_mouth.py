from penman import Graph, load
from typing import List
from evaluation.util import get_node_by_name
from evaluation.graph_matcher import equals_modulo_isomorphy


EDGE_LABELS_AND_REIFICATIONS = [
    [":purpose", "have-purpose-91"],
    [":beneficiary", "benefit-01"],
    [":duration", "last-01"],
    [":topic", "concern-02"],
    [":accompanier", "accompany-01"],
    [":manner", "have-manner-91"],
    [":instrument", "have-instrument-91"],
    [":location", "be-located-at-91"],
    [":time", "be-temporally-at-91"],
]

IDS_FOR_EXACT_MATCH = [
    "anna_bert.tsv-0",
    "anna_bert.tsv-1",
    "anna_bert.tsv-2",
    "anna_bert.tsv-3"
]


def evaluate_berts_mouth(gold_amrs: List[Graph], predicted_amrs: List[Graph]):
    """
    This is the main function for evaluating BERT's Mouth.
    """

    sample_size = 0
    success_count = 0
    for gold, pred in zip(gold_amrs, predicted_amrs):
        sample_size += 1
        if gold.metadata["suppl"] in IDS_FOR_EXACT_MATCH:
            if equals_modulo_isomorphy(gold, pred):
                success_count += 1
            continue
        found_relation = False
        for edge_label, reification in EDGE_LABELS_AND_REIFICATIONS:
            if is_relation_present_in_graph(edge_label, gold, reification):
                found_relation = True
                if is_relation_present_in_graph(edge_label, pred, reification):
                    success_count += 1
                break
        if not found_relation:
            root_label = get_node_by_name(gold.top, gold).target
            if len([inst for inst in pred.instances() if inst.target == root_label]) >= 1:
                success_count += 1

    return success_count, sample_size

def is_relation_present_in_graph(edge_label, graph, reification):
    return len(graph.edges(role=edge_label)) >= 1 \
           or len([inst for inst in graph.instances() if inst.target == reification]) >= 1

def main():
    gold_amrs = load("../corpus/berts_mouth.txt")
    predicted_amrs = load("../amrbart-output/berts_mouth.txt")
    print(evaluate_berts_mouth(gold_amrs, predicted_amrs))


if __name__ == "__main__":
    main()
