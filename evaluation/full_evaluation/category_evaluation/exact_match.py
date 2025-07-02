from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.subcategory_info import is_sanity_check
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.novel_corpus.long_lists import compute_conjunct_counts, compute_generalization_op_counts
from evaluation.util import filter_amrs_for_name


class ExactMatch(CategoryEvaluation):
    """
    We use this only for Structural Generalisation categories, but could be used for other things.
    Checks exact match and Smatch.
    """
    def __init__(self, gold_amrs, predicted_amrs, root_dir,
                 category_metadata, predictions_directory=None, do_error_analysis=False):
        super().__init__(gold_amrs, predicted_amrs, root_dir, category_metadata, predictions_directory, do_error_analysis)

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
        self._get_all_results()
        self._calculate_metrics_and_add_all_rows()
        self.make_smatch_results()
        return self.rows

