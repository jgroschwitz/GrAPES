from typing import List

from penman import load, Graph

from evaluation.corpus_metrics import calculate_edge_prereq_recall_and_sample_size_counts, \
    calculate_node_label_successes_and_sample_size, calculate_subgraph_existence_successes_and_sample_size, \
    compute_smatch_f_from_graph_lists
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.testset.ne_types import get_2_columns_from_tsv_by_id, get_ne_type_successes_and_sample_size
from evaluation.testset.special_entities import get_graphid2labels_from_tsv_file, \
    calculate_special_entity_successes_and_sample_size

EVAL_TYPE_SUCCESS_RATE = "success_rate"
EVAL_TYPE_F1 = "f1"


class CategoryEvaluation:

    def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph], root_dir: str,
                 category_metadata: SubcategoryMetadata):
        self.gold_amrs = gold_amrs
        self.predicted_amrs = predicted_amrs
        self.root_dir = root_dir
        self.corpus_path = f"{self.root_dir}/corpus"
        self.rows = []
        self.category_metadata = category_metadata
        self.print_dataset_name = True  # we want to print the dataset name only on the first metric calculation

    # def set_category_metadata(self, data):
    #     self.category_metadata = data

    def make_and_append_results_row(self, metric_name: str, eval_type: str, metric_results: List):
        """
        :param eval_type: either EVAL_TYPE_SUCCESS_RATE or EVAL_TYPE_F1 (the constants given category_evaluation.py)
        :param metric_name:
        :param metric_results:
        :return:
        """
        new_row = self.make_results_row(metric_name, eval_type, metric_results)
        self.rows.append(new_row)

    def make_results_row(self, metric_name, eval_type, metric_results):
        """
        Include the main dataset name in the result rows only the first time to reduce clutter.
        Args:
            metric_name: Name of the metric such a Edge Recall
            eval_type:
            metric_results: Output of an evaluation, e.g. [successes, sample_size]
        Returns: new row: [display name if new, metric name, eval type, successes, sample_size]

        """
        if self.print_dataset_name:
            ds_name = self.category_metadata #.display_name TODO
            self.print_dataset_name = False  # don't print it next time
        else:
            ds_name = None  # ""
        new_row = [ds_name, metric_name, eval_type] + metric_results
        return new_row

    def make_results_columns_for_edge_recall(self):
        # gold_amrs, predicted_amrs = self.manage_override_amrs(self.category_metadata.override_gold_amrs,
        #                                                       self.category_metadata.override_predicted_amrs)
        self.rows.extend(self.make_results_columns_for_edge_recall_from_graphs(self.gold_amrs, self.predicted_amrs))

    def make_results_columns_for_edge_recall_from_graphs(self, gold_amrs, predicted_amrs):
        prereqs, unlabeled_recalled, labeled_recalled, sample_size = calculate_edge_prereq_recall_and_sample_size_counts(
            self.category_metadata,
            gold_amrs=gold_amrs,
            predicted_amrs=predicted_amrs,
            root_dir=self.root_dir,
        )
        return [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [labeled_recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE,
                                      [unlabeled_recalled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]

    def make_results_column_for_node_recall(self, prereq=False):
        self.rows.append(self.make_results_column_for_node_recall_from_graphs(self.gold_amrs, self.predicted_amrs, prereq))

    def make_results_column_for_node_recall_from_graphs(self,
                                                        gold_amrs: List[Graph],
                                                        predicted_amrs: List[Graph],
                                                        prereq: bool = False):
        success_count, sample_size = calculate_node_label_successes_and_sample_size(
            self.category_metadata,
            gold_amrs=gold_amrs,
            predicted_amrs=predicted_amrs,
            root_dir=self.root_dir,
            prereq=prereq
        )
        metric_label = "Prerequisite" if prereq else self.category_metadata.metric_label
        return self.make_results_row(metric_label, EVAL_TYPE_SUCCESS_RATE, [success_count, sample_size])

    def make_results_for_ne_types(self):
        """
        for named entities
        Returns:

        """
        id2labels = get_2_columns_from_tsv_by_id(f"{self.corpus_path}/{self.category_metadata.tsv}")
        prereq, successes, sample_size = get_ne_type_successes_and_sample_size(
            id2labels,
            self.gold_amrs,
            self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq, sample_size])

    def make_results_for_ne(self):
        id2labels_entities = get_graphid2labels_from_tsv_file(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                                              graph_id_column=self.category_metadata.graph_id_column,
                                                              label_column=self.category_metadata.label_column)
        successes, sample_size = calculate_special_entity_successes_and_sample_size(
            id2labels_entities, self.gold_amrs, self.predicted_amrs, self.category_metadata.subtype)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE,
                                         [successes, sample_size])

    def make_results_for_subgraph(self):
        id2subgraphs = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        recalled, sample_size = calculate_subgraph_existence_successes_and_sample_size(
            id2subgraphs, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])

    def make_smatch_results(self):
        smatch_f1 = compute_smatch_f_from_graph_lists(self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1, [smatch_f1])

    def run_evaluation(self):
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


