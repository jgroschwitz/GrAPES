from typing import List

from penman import Graph

from evaluation.corpus_metrics import compute_smatch_f_from_graph_lists
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata


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

    def make_smatch_results(self):
        smatch = compute_smatch_f_from_graph_lists(self.gold_amrs, self.predicted_amrs)
        smatch_f1 = self.get_f_from_prf(smatch)
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1, [smatch_f1])

    def run_evaluation(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    @staticmethod
    def get_f_from_prf(triple):
        return triple[2]

    def get_gold_filepath(self, dataset_filename=None):
        if dataset_filename is None:
            dataset_filename = self.category_metadata.subcorpus_filename
        return self.root_dir + "/corpus/" + dataset_filename + ".txt"



