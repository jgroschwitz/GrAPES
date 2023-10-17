from typing import List

from penman import load, Graph

from evaluation.corpus_metrics import calculate_edge_prereq_recall_and_sample_size_counts, \
    calculate_node_label_successes_and_sample_size


EVAL_TYPE_SUCCESS_RATE = "success_rate"
EVAL_TYPE_F1 = "f1"


class CategoryEvaluation:

    def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph], parser_name: str, root_dir: str):
        self.dataset_name = None
        self.gold_amrs = gold_amrs
        self.predicted_amrs = predicted_amrs
        self.parser_name = parser_name
        self.root_dir = root_dir
        self.rows = []

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def make_results_column(self, metric_name: str, eval_type: str, metric_results: List):
        """
        If you moved on to a new dataset, then call set_dataset_name() before calling this method.
        :param eval_type: either EVAL_TYPE_SUCCESS_RATE or EVAL_TYPE_F1 (the constants given category_evaluation.py)
        :param metric_name:
        :param metric_results:
        :return:
        """
        if self.dataset_name:
            ds_name = self.dataset_name
            self.dataset_name = None
        else:
            ds_name = ""
        self.rows.append([ds_name, metric_name, eval_type] + metric_results)

    def make_results_columns_for_edge_recall(self, tsv_filename, graph_id_column=0, source_column=1, edge_column=2,
                                             target_column=3, parent_column=None, parent_edge_column=None,
                                             use_sense=False,
                                             override_gold_amrs=None, override_predicted_amrs=None,
                                             first_row_is_header=False):
        """
        Assumes you have called set_dataset_name() before calling this method.
        :param use_sense: default False
        :param tsv_filename:
        :param graph_id_column: default 0
        :param source_column: default 1
        :param edge_column: default 2
        :param target_column: default 3
        :param parent_column: for a second parent
        :param parent_edge_column: for  a second parent
        :return:
        """
        if override_gold_amrs:
            gold_amrs = override_gold_amrs
        else:
            gold_amrs = self.gold_amrs
        if override_predicted_amrs:
            predicted_amrs = override_predicted_amrs
        else:
            predicted_amrs = self.predicted_amrs
        prereqs, unlabeled_recalled, labeled_recalled, sample_size = calculate_edge_prereq_recall_and_sample_size_counts(
            tsv_file_name=tsv_filename,
            gold_amrs=gold_amrs,
            predicted_amrs=predicted_amrs,
            parser_name=self.parser_name,
            root_dir=self.root_dir,
            graph_id_column=graph_id_column,
            source_column=source_column,
            edge_column=edge_column,
            target_column=target_column,
            parent_column=parent_column,
            parent_edge_column=parent_edge_column,
            use_sense=use_sense,
            first_row_is_header=first_row_is_header
        )
        self.make_results_column("Edge recall", EVAL_TYPE_SUCCESS_RATE, [labeled_recalled, sample_size])
        self.make_results_column("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled_recalled, sample_size])
        self.make_results_column("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])

    def make_results_column_for_node_recall(self, tsv_filename, use_sense=False, use_attributes=False,
                                            attribute_label=None, metric_label="Label recall",
                                            override_gold_amrs=None, override_predicted_amrs=None):
        if override_gold_amrs:
            gold_amrs = override_gold_amrs
        else:
            gold_amrs = self.gold_amrs
        if override_predicted_amrs:
            predicted_amrs = override_predicted_amrs
        else:
            predicted_amrs = self.predicted_amrs
        success_count, sample_size = calculate_node_label_successes_and_sample_size(
            tsv_file_name=tsv_filename,
            use_sense=use_sense,
            gold_amrs=gold_amrs,
            predicted_amrs=predicted_amrs,
            parser_name=self.parser_name,
            root_dir=self.root_dir,
            use_attributes=use_attributes,
            attribute_label=attribute_label
        )
        self.make_results_column(metric_label, EVAL_TYPE_SUCCESS_RATE, [success_count, sample_size])

    def get_result_rows(self):
        self._run_all_evaluations()
        return self.rows

    def _run_all_evaluations(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    @staticmethod
    def get_f_from_prf(triple):
        return triple[2]

    def get_prediction_filepath(self, dataset_filename):
        return self.root_dir + "/" + self.parser_name + "-output/" + dataset_filename + ".txt"

    def get_gold_filepath(self, dataset_filename):
        return self.root_dir + "/corpus/" + dataset_filename + ".txt"

    def get_gold_and_pred_for_corpus(self, corpus_name):
        gold = load(self.get_gold_filepath(corpus_name))
        pred = load(self.get_prediction_filepath(corpus_name))
        return gold, pred