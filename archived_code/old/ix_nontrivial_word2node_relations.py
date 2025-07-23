from evaluation.corpus_metrics import calculate_subgraph_existence_successes_and_sample_size
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE

from evaluation.testset.ellipsis import get_ellipsis_success_counts
from evaluation.testset.imperative import get_imperative_success_counts


class NontrivialWord2NodeRelations(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Ellipsis")
        prereqs, recalled, sample_size = get_ellipsis_success_counts(gold_amrs=self.gold_amrs,
                                                                        predicted_amrs=self.predicted_amrs,
                                                                        root_dir=self.root_dir)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])

        self.set_dataset_name("Multinode constants")
        recalled, sample_size = calculate_subgraph_existence_successes_and_sample_size("multinode_constants_filtered.tsv",
                                                                                       self.gold_amrs,
                                                                                       self.predicted_amrs,
                                                                                       self.root_dir)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])

        self.set_dataset_name("Imperatives")
        prereqs, recalled, with_correct_target, sample_size = get_imperative_success_counts(gold_amrs=self.gold_amrs,
                                                                                            predicted_amrs=self.predicted_amrs,
                                                                                            root_dir=self.root_dir)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [with_correct_target, sample_size])
        # self.make_results_column("Marked as imperative", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisite", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])

    def compute_ellipsis_results(self, gold_graphs, predicted_graphs):
        prereqs, recalled, sample_size = get_ellipsis_success_counts(gold_graphs, predicted_graphs, self.root_dir)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]

    def compute_multinode_constants_results(self, gold_graphs, predicted_graphs):
        recalled, sample_size = calculate_subgraph_existence_successes_and_sample_size("multinode_constants_filtered.tsv",
                                                                                       gold_graphs,
                                                                                       predicted_graphs,
                                                                                       self.root_dir)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])]

    def compute_imperative_results(self, gold_graphs, predicted_graphs):
        prereqs, recalled, with_correct_target, sample_size = get_imperative_success_counts(gold_graphs, predicted_graphs, self.root_dir)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [with_correct_target, sample_size]),
                self.make_results_row("Prerequisite", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]