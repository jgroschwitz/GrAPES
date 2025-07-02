from collections import Counter
from typing import List

from penman import Graph

from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size, \
    compute_correctness_counts_from_counter_lists
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.subcategory_info import is_sanity_check, SubcategoryMetadata
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.novel_corpus.long_lists import compute_conjunct_counts, get_all_opi_edges, \
    compute_generalization_op_counts, get_all_unseen_opi_edges
from evaluation.util import copy_graph, remove_edge, get_connected_subgraph_from_node, get_target, with_edge_removed


class ListAccuracy(CategoryEvaluation):

    def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph], root_dir: str,
                 category_metadata: SubcategoryMetadata, predictions_directory=None):
        super().__init__(gold_amrs, predicted_amrs, root_dir, category_metadata, predictions_directory)
        self.error_analysis_dict.update({"recalled_conjuncts": 0,
                                    "conj_total_gold": 0,
                                    "conj_total_predictions": 0,
                                         "correct_opi": [],
                                         "incorrect_opi": []
                                    })
        self.gold_amrs, self.predicted_amrs = self.filter_graphs()


    def _get_all_results(self):
        for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
            graph_id = gold_amr.metadata['id']
            if is_sanity_check(self.category_metadata):
                self.update_error_analysis(graph_id, predicted_amr, gold_amr)
            else:
                self.long_lists_update_error_analysis(graph_id, predicted_amr, gold_amr)
        if not is_sanity_check(self.category_metadata):
            self.compute_generalization_op_counts()

    def _calculate_metrics_and_add_all_rows(self):
        conj_true_predictions = self.error_analysis_dict["recalled_conjuncts"]
        conj_total_gold = self.error_analysis_dict["conj_total_gold"]
        conj_total_predictions = self.error_analysis_dict["conj_total_predictions"]

        print("true predictions:", conj_true_predictions)
        print("total gold:", conj_total_gold)
        print("total predictions:", conj_total_predictions)

        self.make_and_append_results_row("Conjunct recall", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_gold])
        self.make_and_append_results_row("Conjunct precision", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_predictions])

        total_gold = self.error_analysis_dict["unseen_opi_total_gold"]
        total_predictions = self.error_analysis_dict["unseen_opi_total_predictions"]
        true_predictions = self.error_analysis_dict["unseen_opi_true_predictions"]
        self.make_and_append_results_row("Unseen :opi recall", EVAL_TYPE_SUCCESS_RATE,
                                         [true_predictions, total_gold])




    def long_lists_update_error_analysis(self,graph_id, pred, gold):
        """
        Unusually, here we store counts, not just graph IDs, and only one copy of graph IDs for successes and failures
        """
        pred_op_edges = get_all_opi_edges(pred)
        self.error_analysis_dict["conj_total_predictions"] += len(pred_op_edges)

        gold_op_edges = get_all_opi_edges(gold)
        self.error_analysis_dict["conj_total_gold"] += len(gold_op_edges)

        # initialise to all, subtract as we match with gold
        spurious_predictions = len(pred_op_edges)
        recalled_edges = 0
        contains_an_error = False
        for gold_op_edge in gold_op_edges:
            hit = self.gold_edge_has_a_match(gold, gold_op_edge, pred, pred_op_edges)
            if hit:
                # hit
                recalled_edges += 1
                spurious_predictions -= 1
            else:
                # miss
                contains_an_error = True
        if spurious_predictions > 0:
            # predictions that weren't matched with an edge in gold graph
            contains_an_error = True

        if contains_an_error:
            self.add_fail(graph_id)
        else:
            self.add_success(graph_id)

        self.error_analysis_dict["recalled_conjuncts"] += recalled_edges


    def gold_edge_has_a_match(self, gold, gold_op_edge, pred, pred_op_edges):
        gold_graph_without_op_edge = copy_graph(gold)
        remove_edge(gold_graph_without_op_edge, gold_op_edge)
        subgraph = get_connected_subgraph_from_node(get_target(gold_op_edge, gold), gold_graph_without_op_edge)

        return any(
            equals_modulo_isomorphy(subgraph,
                                    get_connected_subgraph_from_node(
                                        get_target(op_edge, pred), with_edge_removed(pred, op_edge)),
                                    match_senses=False,
                                    match_edge_labels=False)
            for op_edge in pred_op_edges)

    def compute_generalization_op_counts(self):
        """

        :param predictions:
        :param golds:
        :return: total_gold, total_predictions, true_predictions
        """
        counter_list_predictions = []
        counter_list_gold = []

        for pred, gold in zip(self.predicted_amrs, self.gold_amrs):
            op_counter_pred = Counter()
            op_counter_gold = Counter()
            counter_list_predictions.append(op_counter_pred)
            counter_list_gold.append(op_counter_gold)

            predicted_unseen = [e.role for e in get_all_unseen_opi_edges(pred)]
            op_counter_pred.update(predicted_unseen)

            gold_unseen = [e.role for e in get_all_unseen_opi_edges(gold)]
            op_counter_gold.update(gold_unseen)

            # update error analysis
            if predicted_unseen != gold_unseen:
                self.error_analysis_dict["incorrect_opi"].append(gold.metadata['id'])
            else:
                self.error_analysis_dict["correct_opi"].append(gold.metadata['id'])

        total_gold, total_predictions, true_predictions = compute_correctness_counts_from_counter_lists(
            counter_list_gold, counter_list_predictions)

        self.error_analysis_dict["unseen_opi_total_gold"] = total_gold
        self.error_analysis_dict["unseen_opi_total_predictions"] = total_predictions
        self.error_analysis_dict["unseen_opi_true_predictions"] = true_predictions


    def long_lists(self):
        conj_total_gold, conj_total_predictions, conj_true_predictions = compute_conjunct_counts(self.gold_amrs,
                                                                                                 self.predicted_amrs)

        self.make_and_append_results_row("Conjunct recall", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_gold])
        self.make_and_append_results_row("Conjunct precision", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_predictions])

        opi_gen_total_gold, opi_gen_total_predictions, opi_gen_true_predictions = compute_generalization_op_counts(
            self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Unseen :opi recall", EVAL_TYPE_SUCCESS_RATE,
                                         [opi_gen_true_predictions, opi_gen_total_gold])

    def long_list_sanity_check(self):
        success, sample_size = compute_exact_match_successes_and_sample_size(self.gold_amrs, self.predicted_amrs,
                                                                             match_edge_labels=False,
                                                                             match_senses=False)
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE, [success, sample_size])
