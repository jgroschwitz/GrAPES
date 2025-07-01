from typing import List

from evaluation.berts_mouth import evaluate_berts_mouth
from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size, compute_smatch_f, \
    compute_smatch_f_from_graph_lists, calculate_subgraph_existence_successes_and_sample_size, \
    calculate_node_label_successes_and_sample_size, calculate_edge_prereq_recall_and_sample_size_counts
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.long_lists import compute_conjunct_counts, compute_generalization_op_counts
from evaluation.pp_attachment import get_pp_attachment_success_counters
from evaluation.structural_generalization import add_sanity_check_suffix, get_exact_match_by_size, size_mappers
from evaluation.testset.ellipsis import get_ellipsis_success_counts
from penman import load, Graph

from evaluation.testset.imperative import get_imperative_success_counts
from evaluation.testset.ne_types import get_2_columns_from_tsv_by_id, get_ne_type_successes_and_sample_size
from evaluation.testset.special_entities import get_graphid2labels_from_tsv_file, \
    calculate_special_entity_successes_and_sample_size
from evaluation.util import filter_amrs_for_name
from evaluation.word_disambiguation import evaluate_word_disambiguation


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

    def make_results_columns_for_edge_recall(self):
        prereqs, unlabeled_recalled, labeled_recalled, sample_size = calculate_edge_prereq_recall_and_sample_size_counts(
            self.category_metadata,
            gold_amrs=self.gold_amrs,
            predicted_amrs=self.predicted_amrs,
            root_dir=self.root_dir,
        )
        rows = [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [labeled_recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE,
                                      [unlabeled_recalled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]
        self.rows.extend(rows)


class NodeRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_column_for_node_recall()
        if self.category_metadata.run_prerequisites:
            self.make_results_column_for_node_recall(prereq=True)
        return self.rows

    def make_results_column_for_node_recall(self, prereq=False):
        success_count, sample_size = calculate_node_label_successes_and_sample_size(
            self.category_metadata,
            gold_amrs=self.gold_amrs,
            predicted_amrs=self.predicted_amrs,
            root_dir=self.root_dir,
            prereq=prereq
        )
        metric_label = "Prerequisite" if prereq else self.category_metadata.metric_label
        row = self.make_results_row(metric_label, EVAL_TYPE_SUCCESS_RATE, [success_count, sample_size])
        self.rows.append(row)


class PPAttachment(CategoryEvaluation):
    def run_evaluation(self):
        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.gold_amrs, self.predicted_amrs)
        return [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]


class PPAttachmentAlone(PPAttachment):
    """
    This is called from evaluate_single_category if you want the PP results but you only have separate output files
    """
    def __init__(self, root_dir: str,
                 category_metadata: SubcategoryMetadata, path_to_predictions_folder):
        super().__init__([], [], root_dir, category_metadata)
        self.get_all_pp_graphs(path_to_predictions_folder)

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


class NERecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_for_ne()
        return self.rows

    def make_results_for_ne(self):
        id2labels_entities = get_graphid2labels_from_tsv_file(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                                              graph_id_column=self.category_metadata.graph_id_column,
                                                              label_column=self.category_metadata.label_column)
        successes, sample_size = calculate_special_entity_successes_and_sample_size(
            id2labels_entities, self.gold_amrs, self.predicted_amrs, self.category_metadata.subtype)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE,
                                         [successes, sample_size])


class SubgraphRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_for_subgraph()
        return self.rows

    def make_results_for_subgraph(self):
        id2subgraphs = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        recalled, sample_size = calculate_subgraph_existence_successes_and_sample_size(
            id2subgraphs, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])


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


class WordDisambiguationRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_for_disambiguation()
        return self.rows

    def make_results_for_disambiguation(self):
        if self.category_metadata.subtype == "hand-crafted":
            fun = evaluate_word_disambiguation
        elif self.category_metadata.subtype == "bert":
            fun = evaluate_berts_mouth
        else:
            raise NotImplementedError(f"subtype {self.category_metadata.subtype} not implemented: must be bert or hand-crafted")
        self.gold_amrs, self.predicted_amrs = filter_amrs_for_name(self.category_metadata.subcorpus_filename,
                                                                   self.gold_amrs, self.predicted_amrs)
        successes, sample_size = fun(self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])


class ExactMatch(CategoryEvaluation):
    def __init__(self, gold_amrs, predicted_amrs, root_dir,
                 category_metadata):
        super().__init__(gold_amrs, predicted_amrs, root_dir, category_metadata)

        #
        if self.need_3s():
            more_golds, more_preds = self.get_3s_amrs()

        self.gold_amrs, self.predicted_amrs = filter_amrs_for_name(
            self.category_metadata.subcorpus_filename,
            self.gold_amrs,
            self.predicted_amrs)

        if self.need_3s():
            self.gold_amrs += more_golds
            self.predicted_amrs += more_preds

    def need_3s(self):
        return self.category_metadata.subcorpus_filename.startswith("deep_recursion_pronouns")

    def get_3s_amrs(self):
        filter_name = add_sanity_check_suffix("deep_recursion_3s") if self.category_metadata.subcorpus_filename.endswith("sanity_check") else "deep_recursion_3s"
        more_golds, more_preds = filter_amrs_for_name(filter_name, self.gold_amrs, self.predicted_amrs)
        return more_golds, more_preds

    def get_results_by_size(self):
        if self.category_metadata.subcorpus_filename in size_mappers:
            return get_exact_match_by_size(self.gold_amrs, self.predicted_amrs, size_mappers[self.category_metadata.subcorpus_filename])
        else:
            return {}

    def run_evaluation(self):
        if self.category_metadata.subcorpus_filename == "long_lists":
            self.long_lists()
        else:
            self.make_success_results_for_structural_generalisation()
        return self.rows

    def make_success_results_for_structural_generalisation(self):
        successes, sample_size = compute_exact_match_successes_and_sample_size(self.gold_amrs, self.predicted_amrs,
                                                                               match_edge_labels=False,
                                                                               match_senses=False)

        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         [successes, sample_size])
        self.make_smatch_results()

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

