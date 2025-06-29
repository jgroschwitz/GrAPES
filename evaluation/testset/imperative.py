from typing import List

from penman import load, Graph
import csv
from evaluation.util import get_node_by_name
from evaluation.file_utils import load_corpus_from_folder



def evaluate_imperatives(gold_amrs, predicted_amrs:List[Graph], id2labels: dict):
    prereq_recalled, recalled, with_correct_imperative_target_count, total = get_imperative_success_counts(id2labels,
                                                                                                           gold_amrs,
                                                                                                           predicted_amrs,
                                                                                                           )

    prereq_recall = prereq_recalled / total
    recall = recalled / total if total > 0 else 1.0
    with_correct_imperative_target_fraction = with_correct_imperative_target_count / total if total > 0 else 1.0

    return prereq_recall, recall, with_correct_imperative_target_fraction


def get_imperative_success_counts(id2labels, gold_amrs, predicted_amrs):
    total = 0
    prereq_recalled = 0
    recalled = 0
    with_correct_imperative_target_count = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels:
            total += len(id2labels[gold_amr.metadata['id']])
            print(id2labels[gold_amr.metadata['id']])
            print(len(id2labels[gold_amr.metadata['id']]))
            for verb_label, imperative_target_edge_label, imperative_target in id2labels[gold_amr.metadata['id']]:
                matching_instances = [instance for instance in predicted_amr.instances() if
                                      instance.target == verb_label]
                if len(matching_instances) >= 1:
                    prereq_recalled += 1
                    with_imperative = [instance for instance in matching_instances if
                                       len(predicted_amr.attributes(source=instance.source, role=":mode",
                                                                    target="imperative")) > 0]
                    if len(with_imperative) >= 1:
                        recalled += 1
                        found_imperative_target = False
                        for instance in with_imperative:
                            for edge in predicted_amr.edges(source=instance.source, role=":" + imperative_target_edge_label):
                                imperative_target_node = get_node_by_name(edge.target, predicted_amr)
                                if imperative_target_node is not None and imperative_target_node.target == imperative_target:
                                    found_imperative_target = True
                        if found_imperative_target:
                            with_correct_imperative_target_count += 1
    return prereq_recalled, recalled, with_correct_imperative_target_count, total


def main():
    print(evaluate_imperatives(parser_name="amparser"))
    print(evaluate_imperatives(parser_name="amrbart"))


if __name__ == "__main__":
    main()