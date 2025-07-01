from collections import Counter
from typing import List, Set

import re

from penman import Graph, decode, encode

from evaluation.util import is_unseen_coord_opi_edge, get_connected_subgraph_from_node, \
    remove_edge, copy_graph, get_target, \
    with_edge_removed, is_opi_edge, get_node_by_name
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.corpus_metrics import compute_precision_recall_f1_from_counter_lists, compute_precision_recall_f1, \
    compute_correctness_counts_from_counter_lists


def evaluate_long_lists(predictions: List[Graph], golds: List[Graph]):
    counter_list_predictions = []
    counter_list_gold = []
    for pred, gold in zip(predictions, golds):
        op_counter_pred = Counter()
        op_counter_gold = Counter()
        counter_list_predictions.append(op_counter_pred)
        counter_list_gold.append(op_counter_gold)
        op_counter_pred.update([e.role for e in get_all_opi_edges(pred)])
        op_counter_gold.update([e.role for e in get_all_opi_edges(gold)])
    op_results = compute_precision_recall_f1_from_counter_lists(counter_list_predictions, counter_list_gold)

    total_gold, total_predictions, true_predictions = compute_conjunct_counts(golds, predictions)
    conjunct_results = compute_precision_recall_f1(true_predictions, total_predictions, total_gold)

    return op_results, conjunct_results


def compute_conjunct_counts(golds, predictions):
    total_predictions = 0
    total_gold = 0
    true_predictions = 0
    for pred, gold in zip(predictions, golds):
        true_predictions_before = true_predictions
        pred_op_edges = get_all_opi_edges(pred)
        gold_op_edges = get_all_opi_edges(gold)
        total_predictions += len(pred_op_edges)
        total_gold += len(gold_op_edges)
        for gold_op_edge in gold_op_edges:
            gold_graph_without_op_edge = copy_graph(gold)
            remove_edge(gold_graph_without_op_edge, gold_op_edge)
            subgraph = get_connected_subgraph_from_node(get_target(gold_op_edge, gold), gold_graph_without_op_edge)

            # FOR DEBUGGING
            # print(encode(subgraph))
            # matches_an_op_child_in_pred = False
            # for op_edge in pred_op_edges:
            #     possible_pred_match = get_connected_subgraph_from_node(
            #                                     get_target(op_edge, pred),
            #                                     with_edge_removed(pred, op_edge))
            #     if equals_modulo_isomorphy(subgraph, possible_pred_match):
            #         matches_an_op_child_in_pred = True
            #         print("Matched!")
            #         print(encode(possible_pred_match))
            # print("\n\n")

            matches_an_op_child_in_pred = any(equals_modulo_isomorphy(subgraph,
                                                                      get_connected_subgraph_from_node(
                                                                          get_target(op_edge, pred),
                                                                          with_edge_removed(pred, op_edge)),
                                                                      match_senses=False,
                                                                      match_edge_labels=False)
                                              for op_edge in pred_op_edges)
            if matches_an_op_child_in_pred:
                true_predictions += 1
            # elif contains_subgraph_modulo_isomorphy(pred, subgraph):
            #     print(encode(pred))
            #     print(encode(subgraph))
        # if true_predictions-true_predictions_before < len(pred_op_edges):
        # print(f"incorrect prediction found? {true_predictions-true_predictions_before} < {len(pred_op_edges)}")
        # print(pred_op_edges)
        # print(encode(pred))
        # print(encode(gold))
        # for op_edge in pred_op_edges:
        #     pred_subgraph = get_connected_subgraph_from_node(
        #                                    get_target(op_edge, pred),
        #                                    )
        #     if not equals_modulo_isomorphy(subgraph,
        #                                ):
        #         matches_an_op_child_in_pred = True
        #         break
        #     else:

        # print(encode(subgraph))
        # if contains_subgraph_modulo_isomorphy(pred, subgraph):
        #     # TODO ignore senses here?
        #     # TODO this whole thing should take multiplicity into account, but at this moment it doesn't
        #     #       Or is the multiplicity solved by the fact that there are no duplicates in the gold?
        #     #       I think it is!
        #     # TODO should we check that each subgraph here is actually an opi child in pred? I think so!
        #     true_predictions += 1
        # print("--------------------\n\n")
    return total_gold, total_predictions, true_predictions


def compute_generalization_op_counts(predictions: List[Graph], golds: List[Graph]):
    """

    :param predictions:
    :param golds:
    :return: total_gold, total_predictions, true_predictions
    """
    counter_list_predictions = []
    counter_list_gold = []
    for pred, gold in zip(predictions, golds):
        op_counter_pred = Counter()
        op_counter_gold = Counter()
        counter_list_predictions.append(op_counter_pred)
        counter_list_gold.append(op_counter_gold)
        op_counter_pred.update([e.role for e in get_all_unseen_opi_edges(pred)])
        op_counter_gold.update([e.role for e in get_all_unseen_opi_edges(gold)])
    return compute_correctness_counts_from_counter_lists(counter_list_predictions, counter_list_gold)


def evaluate_long_lists_generalization(predictions: List[Graph], golds: List[Graph]):
    counter_list_predictions = []
    counter_list_gold = []
    for pred, gold in zip(predictions, golds):
        op_counter_pred = Counter()
        op_counter_gold = Counter()
        counter_list_predictions.append(op_counter_pred)
        counter_list_gold.append(op_counter_gold)
        op_counter_pred.update([e.role for e in get_all_unseen_opi_edges(pred)])
        op_counter_gold.update([e.role for e in get_all_unseen_opi_edges(gold)])
    op_results = compute_precision_recall_f1_from_counter_lists(counter_list_predictions, counter_list_gold)

    total_predictions = 0
    total_gold = 0
    true_predictions = 0
    for pred, gold in zip(predictions, golds):
        true_predictions_before = true_predictions
        pred_op_edges = get_all_unseen_opi_edges(pred)
        gold_op_edges = get_all_unseen_opi_edges(gold)
        total_predictions += len(pred_op_edges)
        total_gold += len(gold_op_edges)
        for gold_op_edge in gold_op_edges:
            gold_graph_without_op_edge = copy_graph(gold)
            remove_edge(gold_graph_without_op_edge, gold_op_edge)
            subgraph = get_connected_subgraph_from_node(get_target(gold_op_edge, gold), gold_graph_without_op_edge)

            # FOR DEBUGGING
            # print(encode(subgraph))
            # matches_an_op_child_in_pred = False
            # for op_edge in pred_op_edges:
            #     possible_pred_match = get_connected_subgraph_from_node(
            #                                     get_target(op_edge, pred),
            #                                     with_edge_removed(pred, op_edge))
            #     if equals_modulo_isomorphy(subgraph, possible_pred_match):
            #         matches_an_op_child_in_pred = True
            #         print("Matched!")
            #         print(encode(possible_pred_match))
            # print("\n\n")

            matches_an_op_child_in_pred = any(equals_modulo_isomorphy(subgraph,
                                                                      get_connected_subgraph_from_node(
                                                                          get_target(op_edge, pred),
                                                                          with_edge_removed(pred, op_edge)),
                                                                      match_senses=False,
                                                                      match_edge_labels=False)
                                              for op_edge in pred_op_edges)
            if matches_an_op_child_in_pred:
                true_predictions += 1
    conjunct_results = compute_precision_recall_f1(true_predictions, total_predictions, total_gold)

    return op_results, conjunct_results


def evaluate_singletons(predictions: List[Graph], golds: List[Graph]):
    for prediction, gold in zip(predictions, golds):
        for instance in gold.instances():
            if not any(pred_inst.target == instance.target for pred_inst in prediction.instances()):
                print(f"{instance} missing!")
        for instance in prediction.instances():
            if not any(gold_inst.target == instance.target for gold_inst in gold.instances()):
                print(f"{instance} predicted inaccurately!")
        for edge in gold.edges():
            if not any(get_simple_edge_string(edge, gold) == get_simple_edge_string(pred_edge, prediction)
                       for pred_edge in prediction.edges()):
                print(f"{get_simple_edge_string(edge, gold)} missing!")
        for edge in prediction.edges():
            if not any(get_simple_edge_string(edge, prediction) == get_simple_edge_string(gold_edge, gold)
                       for gold_edge in gold.edges()):
                print(f"{get_simple_edge_string(edge, prediction)} predicted inaccurately!")
        for attr in gold.attributes():
            if not any(get_simple_attr_string(attr, gold) == get_simple_attr_string(pred_attr, prediction)
                       for pred_attr in prediction.attributes()):
                print(f"{get_simple_attr_string(attr, gold)} missing!")
        for attr in prediction.attributes():
            if not any(get_simple_attr_string(attr, prediction) == get_simple_attr_string(gold_attr, gold)
                       for gold_attr in gold.attributes()):
                print(f"{get_simple_attr_string(attr, prediction)} predicted inaccurately!")
    print(f"Done evaluation {len(predictions)} singletons")


def get_simple_edge_string(edge, graph):
    return get_node_by_name(edge.source, graph).target + " " + edge.role + " " +\
           get_node_by_name(edge.target, graph).target


def get_simple_attr_string(attr, graph):
    return f"{attr.role} {attr.target}"


def get_all_opi_edges(graph: Graph):
    # TODO note that since we use graph.edges() here, this ignores properties! In particular, op_i edges in names.
    #  This seems to be what we want, but may accidentally exclude some "proper conjunct opi edges" if there are
    #  e.g. erroneous quotation marks, I think.
    return [e for e in graph.edges() if is_opi_edge(e)]


def get_all_unseen_opi_edges(graph: Graph):
    return [e for e in graph.edges() if is_unseen_coord_opi_edge(e)]


def main():
    graph1 = decode("(e / eat-01 :ARG0 (p / person :ARG0-of (c / cook-01)) :ARG1 (f2 / food))")
    graph2 = decode("(e / eat-01 :ARG0 (p / person :ARG0-of (c / cook-01)) :ARG1 (f2 / food))")
    graph3 = decode("(a / and :op1 (v / vanilla) :op2 (c / chocolate) :op3 (b / banana :mod (s / small)))")
    print(equals_modulo_isomorphy(graph1, graph2))
    print(get_all_opi_edges(graph3))


if __name__ == "__main__":
    main()
