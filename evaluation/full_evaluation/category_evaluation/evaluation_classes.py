from evaluation.novel_corpus.berts_mouth import evaluate_berts_mouth
from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size,  calculate_subgraph_existence_successes_and_sample_size, \
    calculate_node_label_successes_and_sample_size, calculate_edge_prereq_recall_and_sample_size_counts
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata, is_sanity_check
from evaluation.novel_corpus.long_lists import compute_conjunct_counts, compute_generalization_op_counts
from evaluation.novel_corpus.pp_attachment import get_pp_attachment_success_counters
from evaluation.novel_corpus.structural_generalization import get_exact_match_by_size, size_mappers, \
    add_sanity_check_suffix
from evaluation.testset.ellipsis import get_ellipsis_success_counts
from penman import load

from evaluation.testset.imperative import get_imperative_success_counts
from evaluation.testset.ne_types import get_2_columns_from_tsv_by_id, get_ne_type_successes_and_sample_size
from evaluation.testset.special_entities import get_graphid2labels_from_tsv_file, \
    calculate_special_entity_successes_and_sample_size
from evaluation.util import filter_amrs_for_name
from evaluation.novel_corpus.word_disambiguation import evaluate_word_disambiguation


class EdgeRecall(CategoryEvaluation):
    def run_evaluation(self):
        try:
            self.make_results()
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

    def make_results(self):
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
        self.make_results()
        if self.category_metadata.run_prerequisites:
            self.make_results(prereq=True)
        return self.rows

    def make_results(self, prereq=False):
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
    def __init__(self, gold_amrs, predicted_amrs, root_dir, info, predictions_directory=None):
        """
        Pragmatic attachments of ambiguous PPs
        PP Attachments come from multiple files, so if they're not already in the given graphs, we try to get them.
        """
        super().__init__(gold_amrs, predicted_amrs, root_dir, info, predictions_directory)
        # if we read in the unused PP directory instead of the whole full_cprpus, replace it with the real ones
        # These have ids pp_attachment_n
        if self.gold_amrs[0].metadata['id'].startswith(self.category_metadata.subcorpus_filename):
            print("Reading in additional files")
            self.gold_amrs, self.predicted_amrs = self.get_additional_graphs(read_in=True)
        if len(self.gold_amrs) != len(self.predicted_amrs) or len(self.gold_amrs) ==0:
            raise Exception("Different number of AMRs or 0")

    def make_results(self):
        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.gold_amrs, self.predicted_amrs)
        assert sample_size>0, "No results found!"
        rows = [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]
        self.rows.extend(rows)


class NETypeRecall(CategoryEvaluation):
    """Identifying named entity types"""
    def make_results(self):
        id2labels = get_2_columns_from_tsv_by_id(f"{self.corpus_path}/{self.category_metadata.tsv}")
        prereq, successes, sample_size = get_ne_type_successes_and_sample_size(
            id2labels,
            self.gold_amrs,
            self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq, sample_size])


class NERecall(CategoryEvaluation):
    """Correctly creating attributes for named entities, such as the components of a name"""
    def make_results(self):
        id2labels_entities = get_graphid2labels_from_tsv_file(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                                              graph_id_column=self.category_metadata.graph_id_column,
                                                              label_column=self.category_metadata.label_column)
        successes, sample_size = calculate_special_entity_successes_and_sample_size(
            id2labels_entities, self.gold_amrs, self.predicted_amrs, self.category_metadata.subtype)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE,
                                         [successes, sample_size])


class SubgraphRecall(CategoryEvaluation):
    """For multinode word meanings like "teacher" = person <- teach-01"""
    def make_results(self):
        id2subgraphs = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        recalled, sample_size = calculate_subgraph_existence_successes_and_sample_size(
            id2subgraphs, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])


class EllipsisRecall(CategoryEvaluation):
    """Find two instances of a subgraph where one is elided in the sentence"""
    def make_results(self):
        id2labels = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        prereqs, recalled, sample_size = get_ellipsis_success_counts(
            id2labels, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])


class ImperativeRecall(CategoryEvaluation):
    def make_results(self):
        id2labels = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv, columns=[1,2,3])
        prereqs, recalled, with_correct_target, sample_size = get_imperative_success_counts(id2labels,
                                                                                            gold_amrs=self.gold_amrs,
                                                                                            predicted_amrs=self.predicted_amrs,
                                                                                            )
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [with_correct_target, sample_size])
        # self.make_results_column("Marked as imperative", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisite", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])


class WordDisambiguationRecall(CategoryEvaluation):
    def make_results(self):
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
                 category_metadata, predictions_directory=None):
        super().__init__(gold_amrs, predicted_amrs, root_dir, category_metadata, predictions_directory)

        self.is_sanity_check = is_sanity_check(self.category_metadata)

        gold_amrs, predicted_amrs = filter_amrs_for_name(
            self.category_metadata.subcorpus_filename,
            self.gold_amrs,
            self.predicted_amrs)

        # Add extra graphs for deep_recursion_pronouns
        if self.extra_subcorpus_filenames is not None:
            read_in = len(self.gold_amrs) == 100  # exactly 100 graphs in deep_recursion_pronouns.txt.
            more_gold, more_pred =  self.get_additional_graphs(read_in=read_in)
            gold_amrs += more_gold
            predicted_amrs += more_pred

        self.gold_amrs = gold_amrs
        self.predicted_amrs = predicted_amrs

    def run_evaluation(self):
        if self.category_metadata.subcorpus_filename == "long_lists":
            self.long_lists()
        elif self.category_metadata.subcorpus_filename.startswith("long_lists"):
            self.long_list_sanity_check()
        else:
            self.make_results()
        return self.rows

    def make_results(self):
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

