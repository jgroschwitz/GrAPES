import pickle
import sys
from typing import List

import penman
from penman import Graph

from evaluation.corpus_metrics import compute_smatch_f_from_graph_lists, graph_is_in_ids
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.novel_corpus.structural_generalization import size_mappers, get_exact_match_by_size
from evaluation.util import filter_amrs_for_name

EVAL_TYPE_SUCCESS_RATE = "success_rate"
EVAL_TYPE_F1 = "f1"
EVAL_TYPE_NONE = 1
EVAL_TYPE_NA = 0


class CategoryEvaluation:

    def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph], root_dir: str,
                 category_metadata: SubcategoryMetadata, predictions_directory=None):
        self.gold_amrs = gold_amrs
        self.predicted_amrs = predicted_amrs
        self.root_dir = root_dir
        self.corpus_path = f"{self.root_dir}/corpus"
        self.rows = []
        self.category_metadata = category_metadata
        self.print_dataset_name = True  # we want to print the dataset name only on the first metric calculation
        self.extra_subcorpus_filenames = category_metadata.extra_subcorpus_filenames
        self.predictions_directory = predictions_directory

        self.error_analysis_dict = {"correct_ids": [], "incorrect_ids": []}

        if self.category_metadata.run_prerequisites:
            self.error_analysis_dict.update({"correct_prereqs": [], "incorrect_prereqs": []})
        if self.measure_unlabelled_edges():
            self.error_analysis_dict.update({"correct_unlabelled": [],"incorrect_unlabelled": []})


    @staticmethod
    def measure_unlabelled_edges():
        # only true for EdgeRecall
        return False

    def run_evaluation(self):
        self._get_all_results()
        self._calculate_metrics_and_add_all_rows()
        return self.rows

    def get_additional_graphs(self, read_in):
        """
        If there are additional graphs required by this category, we can read them in or filter them from the larger set.
        :param: read_in: if True, read them in from a file, otherwise filter them from the stored corpora
        """
        if self.extra_subcorpus_filenames is None:
            raise ValueError("extra_subcorpus_filenames is not defined")
        if not read_in:
            filtered_golds = []
            filtered_preds = []
            for name in self.extra_subcorpus_filenames:
                more_golds, more_preds = filter_amrs_for_name(name, self.gold_amrs, self.predicted_amrs)
                filtered_golds += more_golds
                filtered_preds += more_preds
            return filtered_golds, filtered_preds
        else:
            if self.predictions_directory is not None:
                extra_predictions = []
                extra_golds = []
                for filename in self.extra_subcorpus_filenames:
                    print("reading in", filename)
                    extra_predictions += penman.load(f"{self.predictions_directory}/{filename}.txt")
                    extra_golds += penman.load(f"{self.corpus_path}/subcorpora/{filename}.txt")
                return extra_golds, extra_predictions
            else:
                raise NotImplementedError("Can't get additional graphs without predictions directory")


    def make_and_append_results_row(self, metric_name: str, eval_type: str, metric_results: List):
        """
        :param eval_type: either EVAL_TYPE_SUCCESS_RATE or EVAL_TYPE_F1 (the constants given category_evaluation.py)
        :param metric_name:
        :param metric_results:
        """
        new_row = self.make_results_row(metric_name, eval_type, metric_results)
        self.rows.append(new_row)

    def make_results_row(self, metric_name, eval_type, metric_results):
        """
        Include the main dataset name in the result rows only the first time to reduce clutter.
        Args:
            metric_name: Name of the metric such a Edge Recall
            eval_type: used to guide printing
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

    @staticmethod
    def make_empty_row(category_name="", metric_name="-"):
        return [category_name, metric_name, EVAL_TYPE_NA] + []

    def make_smatch_results(self):
        print("running Smatch...")
        smatch = compute_smatch_f_from_graph_lists(self.gold_amrs, self.predicted_amrs)
        smatch_f1 = self.get_f_from_prf(smatch)
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1, [smatch_f1])

    def get_results_by_size(self):
        """Split up the generalisation by size as marked in corpora.
        Currently just used for structural generalisation"""
        if self.category_metadata.subcorpus_filename in size_mappers:
            return get_exact_match_by_size(self.gold_amrs, self.predicted_amrs, size_mappers[self.category_metadata.subcorpus_filename])
        else:
            return {}

    def make_results(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    # def run_evaluation(self):
    #     self.make_results()
    #     return self.rows

    @staticmethod
    def get_f_from_prf(triple):
        return triple[2]

    def get_gold_filepath(self, dataset_filename=None):
        if dataset_filename is None:
            dataset_filename = self.category_metadata.subcorpus_filename
        return self.root_dir + "/corpus/" + dataset_filename + ".txt"

    def filter_graphs(self):
        """
        Filter by TSV or id containing info.subcorpus_filename
        If neither works (neither true or we get 0 graphs this way) just return the whole stored graph lists.
        Returns: filtered_golds, filtered_predicted
        """
        filtered_golds = []
        filtered_preds = []
        if self.category_metadata.tsv is not None:
            ids = read_label_tsv(self.root_dir, self.category_metadata.tsv, graph_id_column=self.category_metadata.graph_id_column).keys()
            for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
                if graph_is_in_ids(gold_amr, ids):
                    filtered_golds.append(gold_amr)
                    filtered_preds.append(predicted_amr)
        elif self.category_metadata.subcorpus_filename is not None:
            filtered_golds, filtered_preds = filter_amrs_for_name(self.category_metadata.subcorpus_filename, self.gold_amrs, self.predicted_amrs, fail_ok=True)
            if self.category_metadata.extra_subcorpus_filenames is not None:
                for name in self.category_metadata.extra_subcorpus_filenames:
                    more_gold, more_preds = filter_amrs_for_name(name, self.gold_amrs, self.predicted_amrs,
                                                                          fail_ok=False)
                    filtered_golds.extend(more_gold)
                    filtered_preds.extend(more_preds)
        else:
            print("No filtering done")
            filtered_golds = self.gold_amrs
            filtered_preds = self.predicted_amrs
        if len(filtered_golds) == 0:
            print("WARNING: filtering gave us 0 graphs! Returning all", file=sys.stderr)
            filtered_golds = self.gold_amrs
            filtered_preds = self.predicted_amrs
        return filtered_golds, filtered_preds

    def get_all_gold_ids(self):
        filtered_gold, _ = self.filter_graphs()
        return [gold.metadata["id"] for gold in filtered_gold]

    def read_tsv(self):
        return read_label_tsv(self.root_dir, self.category_metadata.tsv)

    def dump_error_analysis_pickle(self):
        with open(f"{self.root_dir}/error_analysis/{self.category_metadata.name}.pickle", "wb") as f:
            pickle.dump(self.error_analysis_dict, f)

    def _calculate_metrics_and_add_all_rows(self):

        success_count = len(self.error_analysis_dict["correct_ids"])
        sample_size = success_count + len(self.error_analysis_dict["incorrect_ids"])
        ret = [success_count, sample_size]
        self.rows.append(self.make_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE,
                                               [success_count, sample_size]))
        if self.measure_unlabelled_edges():
            unlabelled_success_count = len(self.error_analysis_dict["correct_unlabelled"])
            ret.append(unlabelled_success_count)
            self.rows.append(self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE,
            [unlabelled_success_count, sample_size]))
        if self.category_metadata.run_prerequisites:
            prereq_success_count = len(self.error_analysis_dict["correct_prereqs"])
            ret.append(prereq_success_count)
            self.rows.append(self.make_results_row(
                "Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq_success_count, sample_size]))
        print("Metrics:", ret)
        self.dump_error_analysis_pickle()
        return ret

    def get_predictions_for_comparison(self, predicted_amr):
        """Default is just the graph"""
        return predicted_amr

    def _get_all_results(self):
        """
        Loops through graphs and updates error analysis record
        """
        print("Using new method")

        if self.category_metadata.tsv is not None:
            # read in the TSV to get the targets for comparison
            id2labels = self.read_tsv()
            for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
                graph_id = gold_amr.metadata['id']
                if graph_id in id2labels:
                    # occasionally we need something other than just the predicted graph for comparison
                    predictions_for_comparison = self.get_predictions_for_comparison(predicted_amr)
                    for target in id2labels[graph_id]:
                        # update results for this item
                        # if we have a TSV, update_error_analysis is per item in the TSV
                        self.update_error_analysis(graph_id, predictions_for_comparison,
                                                   target)
        else:
            for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
                graph_id = gold_amr.metadata['id']
                # if no TSV, update_error_analysis is per graph pair
                self.update_error_analysis(graph_id, predicted_amr, gold_amr)

    def update_error_analysis(self, graph_id, predictions_for_comparison, target):
        """
        Default: exact match, modulo edge labels and senses
        Args:
            graph_id:
            predictions_for_comparison:
            target:

        Returns:

        """
        print("Running exact match (default)")
        if equals_modulo_isomorphy(target, predictions_for_comparison, match_edge_labels=False, match_senses=False):
            self.add_success(graph_id)
        else:
            self.add_fail(graph_id)

    def add_prereq_success(self, graph_id):
        self.error_analysis_dict["correct_prereqs"].append(graph_id)

    def add_prereq_fail(self, graph_id):
        self.error_analysis_dict["incorrect_prereqs"].append(graph_id)

    def add_success(self, graph_id):
        self.error_analysis_dict["correct_ids"].append(graph_id)

    def add_fail(self, graph_id):
        self.error_analysis_dict["incorrect_ids"].append(graph_id)

    def add_unlabelled_success(self, graph_id):
        self.error_analysis_dict["correct_unlabelled"].append(graph_id)

    def add_unlabelled_fail(self, graph_id):
        self.error_analysis_dict["incorrect_unlabelled"].append(graph_id)
