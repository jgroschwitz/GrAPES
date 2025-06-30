from typing import List

from evaluation.berts_mouth import evaluate_berts_mouth
from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size, compute_smatch_f, \
    compute_smatch_f_from_graph_lists
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

class NodeRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results_column_for_node_recall()
        if self.category_metadata.run_prerequisites:
            self.make_results_column_for_node_recall(prereq=True)
        return self.rows

class PPAttachment(CategoryEvaluation):

    def run_evaluation(self):

        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.gold_amrs, self.predicted_amrs)
        return [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]


class PPAttachmentAlone(PPAttachment):

    def __init__(self, parser_name: str, root_dir: str,
                 category_metadata: SubcategoryMetadata, path_to_predictions_folder):
        super().__init__([], [], parser_name, root_dir, category_metadata)
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
    def __init__(self, gold_amrs, predicted_amrs, parser_name, root_dir,
                 category_metadata):
        super().__init__(gold_amrs, predicted_amrs, parser_name, root_dir,
                 category_metadata)

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

        smatch_f1 = compute_smatch_f_from_graph_lists(self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1, [smatch_f1])

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

        # self.make_results_column(":opi f1", EVAL_TYPE_F1, [self.get_f_from_prf(op_f1)])
        # self.make_results_column("Conjunct f1", EVAL_TYPE_F1, [self.get_f_from_prf(conjunct_f1)])
        # self.make_results_column("Unseen :opi f1", EVAL_TYPE_F1, [self.get_f_from_prf(generalization_op_f1)])
        # self.make_results_column("Conjunct f1 for unseen :opi", EVAL_TYPE_F1,
        #                          [self.get_f_from_prf(generalization_conjunct_f1)])

    def long_list_sanity_check(self):
        success, sample_size = compute_exact_match_successes_and_sample_size(self.gold_amrs, self.predicted_amrs,
                                                                             match_edge_labels=False,
                                                                             match_senses=False)
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE, [success, sample_size])

class StructuralGeneralisation(ExactMatch):
    def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph],
                 gold_sanity_check: List[Graph], predicted_sanity_check: List[Graph], parser_name: str, root_dir: str,
                 category_metadata: SubcategoryMetadata, path_to_predictions_folder: str):
        super().__init__(gold_amrs, predicted_amrs, parser_name, root_dir,
                 category_metadata)
        self.gold_sanity_check = gold_sanity_check
        self.predicted_sanity_check = predicted_sanity_check
        self.path_to_predictions_folder = path_to_predictions_folder
        self.third_person_corpus = "deep_recursion_3s"

    def get_3s_predictions_path(self):
        return f"{self.path_to_predictions_folder}/{self.third_person_corpus}.txt"

    def get_predictions_path(self):
        return f"{self.path_to_predictions_folder}/{self.category_metadata.subcorpus_filename}.txt"

    def get_gold_path(self):
        return f"{self.root_dir}/corpus/subcorpora/{self.category_metadata.subcorpus_filename}.txt"

    def get_3s_gold_path(self):
        return f"{self.root_dir}/corpus/subcorpora/{self.third_person_corpus}.txt"

    def get_smatch(self, gold_path: str = None, predicted_path: str = None):

        if gold_path is None:
            gold_path = self.get_gold_path()
        if predicted_path is None:
            predicted_path = self.get_predictions_path()
        smatch_results = compute_smatch_f(gold_path, predicted_path)
        return smatch_results

    def run_evaluation(self):

        if self.category_metadata.subcorpus_filename == "deep_recursion_pronouns":
            self.pronouns()
        elif self.category_metadata.subcorpus_filename == "long_lists":
            self.long_lists()
        else:
            self.make_success_results_for_structural_generalisation()

        # reset for sanity check
        self.print_dataset_name = True
        self.category_metadata.display_name = "Sanity check"
        self.gold_amrs = self.gold_sanity_check
        self.predicted_amrs = self.predicted_sanity_check
        self.category_metadata.subcorpus_filename = add_sanity_check_suffix(self.category_metadata.subcorpus_filename)
        self.third_person_corpus = add_sanity_check_suffix(self.third_person_corpus)

        if self.category_metadata.subcorpus_filename == add_sanity_check_suffix("deep_recursion_pronouns"):
            self.pronouns()
        elif self.category_metadata.subcorpus_filename == add_sanity_check_suffix("long_lists"):
            self.long_list_sanity_check()
        else:
            self.make_success_results_for_structural_generalisation()

        return self.rows


    def pronouns(self):
        assert len(self.gold_amrs) == len(self.predicted_amrs)
        successes, sample_size = compute_exact_match_successes_and_sample_size(self.gold_amrs, self.predicted_amrs,
                                                                               match_edge_labels=False,
                                                                               match_senses=False)

        smatch_results = self.get_smatch()
        extra_predictions =  load(self.get_3s_predictions_path())
        extra_gold =  load(self.get_3s_gold_path())
        assert len(extra_predictions) == len(extra_gold)
        successes_3s, sample_size_3s = compute_exact_match_successes_and_sample_size(extra_gold, extra_predictions,
                                                                               match_edge_labels=False,
                                                                               match_senses=False)
        smatch_results_3s = self.get_smatch(gold_path=self.get_3s_gold_path(),
                                            predicted_path=self.get_3s_predictions_path())

        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE,
                                         [successes + successes_3s, sample_size + sample_size_3s])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         # taking the average for smatch (not exactly correct, since this overvalues
                                         # the larger corpus, but the sizes should be close enough.
                                         [(self.get_f_from_prf(smatch_results)+ self.get_f_from_prf(smatch_results_3s)) / 2])

