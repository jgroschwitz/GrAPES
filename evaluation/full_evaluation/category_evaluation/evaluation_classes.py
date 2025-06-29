from typing import List

from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.pp_attachment import get_pp_attachment_success_counters
from evaluation.testset.ellipsis import get_ellipsis_success_counts
from penman import load, Graph

from evaluation.testset.imperative import get_imperative_success_counts


class EdgeRecall(CategoryEvaluation):

    def run_evaluation(self):

        try:
            self.make_results_columns_for_edge_recall()
        except IndexError as e:
            if self.category_metadata.subcorpus_filename == "unbounded_dependencies":
                print("Check that corpus/unbounded_dependencies.tsv has 66 rows")
                print("Something may have gone wrong in extending the GrAPES testset with PTB data")
            raise e
        except FileNotFoundError as e:
            if self.category_metadata.subcorpus_filename == "unbounded_dependencies":
                print("Check that corpus/unbounded_dependencies.tsv exists")
                print("Something may have gone wrong in extending the GrAPES testset with PTB data")
            raise e
        return self.rows

class NodeRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_column_for_node_recall()
        if self.category_metadata.run_prerequisites:
            self.make_results_column_for_node_recall(prereq=True)
        return self.rows

class PPAttachment(CategoryEvaluation):

    def __init__(self, parser_name: str, root_dir: str,
                 category_metadata: SubcategoryMetadata, path_to_predictions_folder):
        super().__init__([], [], parser_name, root_dir, category_metadata)
        self.get_all_pp_graphs(path_to_predictions_folder)


    def run_evaluation(self):

        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.gold_amrs, self.predicted_amrs)
        return [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]

    def get_all_pp_graphs(self, path_to_predictions_folder):
        """
        Assumes all predictions are in the same folder and have the same names as the golds
        Args:
            path_to_predictions_folder: path to the parser output AMR files such as see_with.txt
        updates empty self.gold_amrs and self.predicted_amrs with concatenated gold AMRs, concatenated predicted AMRs
        """
        print("Concatenating PP files")
        for filename in ["see_with", "read_by", "bought_for", "keep_from", "give_up_in"]:
            self.gold_amrs += load(f"{self.root_dir}/corpus/subcorpora/{filename}.txt")
            self.predicted_amrs += load(f"{path_to_predictions_folder}/{filename}.txt")
        assert len(self.gold_amrs) == len(self.predicted_amrs) and len(self.gold_amrs) > 0

class NETypeRecall(CategoryEvaluation):

    def run_evaluation(self):
        self.make_results_for_ne_types()
        return self.rows

class NERecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_for_ne()
        return self.rows

class SubgraphRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_for_subgraph()
        return self.rows

class EllipsisRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_for_ellipsis()
        return self.rows

    def make_results_for_ellipsis(self):
        id2labels = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        prereqs, recalled, sample_size = get_ellipsis_success_counts(
            id2labels, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])

class ImperativeRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_for_imperative()
        return self.rows

    def make_results_for_imperative(self):
        id2labels = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv, columns=[1,2,3])
        prereqs, recalled, with_correct_target, sample_size = get_imperative_success_counts(id2labels,
                                                                                            gold_amrs=self.gold_amrs,
                                                                                            predicted_amrs=self.predicted_amrs,
                                                                                            )
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [with_correct_target, sample_size])
        # self.make_results_column("Marked as imperative", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisite", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])
