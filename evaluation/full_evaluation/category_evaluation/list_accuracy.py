from collections import Counter
from typing import List

from penman import Graph

from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size, \
    compute_correctness_counts_from_counter_lists
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, Results, IDResults
from evaluation.full_evaluation.category_evaluation.subcategory_info import is_sanity_check, SubcategoryMetadata
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.novel_corpus.long_lists import compute_conjunct_counts, get_all_opi_edges, \
    compute_generalization_op_counts, get_all_unseen_opi_edges
from evaluation.util import copy_graph, remove_edge, get_connected_subgraph_from_node, get_target, with_edge_removed

OPi = "unseen_opi"

class ListAccuracy(CategoryEvaluation):
    """ For the Long Lists category. Note that the Sanity Check uses ExactMatch instead."""

    def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph], root_dir: str,
                 category_metadata: SubcategoryMetadata, predictions_directory=None, do_error_analysis=False):
        """
        We add space for storing counts to the error analysis because (a) there are a lot of edges per graph,
        and we don't want a copy for each mistake, and (b) we also want to calculate precision.
        """
        super().__init__(gold_amrs, predicted_amrs, root_dir, category_metadata, predictions_directory, do_error_analysis)
        self.gold_amrs, self.predicted_amrs = self.filter_graphs()
        self.results = ListResults()

    def _get_all_results(self):
        """
        Runs both conjunct recall and precision and the unseen opi recall.
        """
        for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
            self.update_results(gold_amr, predicted_amr)
        self.compute_generalization_op_counts()

    def _calculate_metrics_and_add_all_rows(self):
        conj_true_predictions = self.get_success_count()
        conj_total_gold = self.results.get_total_gold()
        conj_total_predictions = self.results.get_total_predictions()

        self.make_and_append_results_row("Conjunct recall", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_gold])
        self.make_and_append_results_row("Conjunct precision", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_predictions])

        # unseen opis
        opi_total_gold = self.results.get_total_gold(OPi)
        true_predictions = self.get_success_count(OPi)
        self.make_and_append_results_row("Unseen :opi recall", EVAL_TYPE_SUCCESS_RATE,
                                         [true_predictions, opi_total_gold])

    def update_results(self, gold_amr, predicted_amr, target=None, predictions_for_comparison=None):
        """
        Unusually, here we store counts, not just graph IDs, and only one copy of graph IDs for successes and failures
        """
        pred_op_edges = get_all_opi_edges(predicted_amr)
        # self.error_analysis_dict["conj_total_predictions"] += len(pred_op_edges)
        self.results.total_predictions_conjuncts += len(pred_op_edges)

        gold_op_edges = get_all_opi_edges(gold_amr)
        # self.error_analysis_dict["conj_total_gold"] += len(gold_op_edges)
        self.results.total_gold_conjuncts += len(gold_op_edges)

        # initialise to all, subtract as we match with gold
        spurious_predictions = len(pred_op_edges)
        recalled_edges = 0
        contains_an_error = False
        for gold_op_edge in gold_op_edges:
            hit = self.gold_edge_has_a_match(gold_amr, gold_op_edge, predicted_amr, pred_op_edges)
            if hit:
                recalled_edges += 1
                spurious_predictions -= 1
            else:
                contains_an_error = True
        if spurious_predictions > 0:
            # predictions that weren't matched with an edge in gold graph
            contains_an_error = True

        if contains_an_error:
            # self.add_fail(graph_id)
            self.add_fail(gold_amr, predicted_amr)
        else:
            self.add_success(gold_amr, predicted_amr)
            #self.add_success(graph_id)

        # self.error_analysis_dict["recalled_conjuncts"] += recalled_edges
        setattr(self.results, self.results.make_success_key(), self.get_success_count() + recalled_edges)


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
                self.add_fail(gold, pred, OPi)
            else:
                self.add_success(gold, pred, OPi)

        total_gold, total_predictions, true_predictions = compute_correctness_counts_from_counter_lists(
            counter_list_gold, counter_list_predictions)

        self.results.total_gold_unseen_opi = total_gold
        self.results.total_predictions_unseen_opi = total_predictions
        self.results.correct_unseen_opi = true_predictions



class ListResults(IDResults):
    def __init__(self):
        additional_fields = [OPi]
        super().__init__(additional_fields=additional_fields, default_field="conjuncts")

        for field in additional_fields + [self.default_field]:
            setattr(self, f"correct_{field}", 0)
            for corpus in ["gold", "predictions"]:
                setattr(self, f"total_{corpus}_{field}", 0)

    def get_success_count(self, field=None):
        return getattr(self, self.make_success_key(field))

    def get_failure_count(self, field=None):
        """
        Returns the sum of the type 1 and type 2 errors
        """
        if field is None:
            field = self.default_field
        successes = self.get_success_count(field)
        type_1_errors = getattr(self, f"total_predictions_{field}") - successes
        type_2_errors = getattr(self, f"total_gold_{field}") - successes
        return type_1_errors + type_2_errors

    def get_total_gold(self, field=None):
        if field is None:
            field = self.default_field
        return getattr(self, f"total_gold_{field}")

    def get_total_predictions(self, field=None):
        if field is None:
            field = self.default_field
        return getattr(self, f"total_predictions_{field}")

