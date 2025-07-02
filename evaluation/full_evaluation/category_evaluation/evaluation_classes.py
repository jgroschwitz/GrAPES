from evaluation.graph_matcher import check_fragment_existence
from evaluation.novel_corpus.berts_mouth import evaluate_berts_mouth
from evaluation.corpus_metrics import calculate_subgraph_existence_successes_and_sample_size
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.novel_corpus.pp_attachment import get_pp_attachment_success_counters
from evaluation.testset.ellipsis import get_ellipsis_success_counts

from evaluation.testset.imperative import get_imperative_success_counts
from evaluation.util import filter_amrs_for_name
from evaluation.novel_corpus.word_disambiguation import evaluate_word_disambiguation


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


class SubgraphRecall(CategoryEvaluation):
    """For multinode word meanings like "teacher" = person <- teach-01"""
    def make_results(self):
        id2subgraphs = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        recalled, sample_size = calculate_subgraph_existence_successes_and_sample_size(
            id2subgraphs, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])

    def update_results(self, gold_amr, predicted_amr, label, predictions_for_comparison=None):
        if check_fragment_existence(label, predicted_amr):
            self.add_success(gold_amr, predicted_amr)
        else:
            self.add_fail(gold_amr, predicted_amr)


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


