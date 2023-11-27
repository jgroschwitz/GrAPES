from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1

from evaluation.testset.ne_types import get_ne_type_successes_and_sample_size

class EntityClassificationAndLinking(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Types of seen named entities")
        prereq, successes, sample_size = get_ne_type_successes_and_sample_size(self.root_dir+"/corpus/seen_ne_types_test.tsv",
                                                                       self.gold_amrs,
                                                                       self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq, sample_size])

        self.set_dataset_name("Types of unseen named entities")
        prereq, successes, sample_size = get_ne_type_successes_and_sample_size(self.root_dir+"/corpus/unseen_ne_types_test.tsv",
                                                                          self.gold_amrs,
                                                                            self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq, sample_size])

        self.set_dataset_name("Seen and/or easy wiki links")
        self.make_results_column_for_node_recall("seen_andor_easy_wiki_test_data.tsv", use_sense=True,
                                                 use_attributes=True,
                                                 attribute_label=":wiki",
                                                 metric_label="Recall")

        self.set_dataset_name("Hard unseen wiki links")
        self.make_results_column_for_node_recall("hard_wiki_test_data.tsv", use_sense=True,
                                                 use_attributes=True,
                                                    attribute_label=":wiki",
                                                    metric_label="Recall")

