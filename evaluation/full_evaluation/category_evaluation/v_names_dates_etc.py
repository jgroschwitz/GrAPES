from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1
from evaluation.testset.special_entities import calculate_name_successes_and_sample_size, \
    calculate_date_successes_and_sample_size, calculate_special_entity_successes_and_sample_size


class NamesDatesEtc(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Seen names")
        successes, sample_size = calculate_name_successes_and_sample_size(self.root_dir + "/corpus/seen_names.tsv",
                                                                          self.gold_amrs,
                                                                          self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])

        self.set_dataset_name("Unseen names")
        successes, sample_size = calculate_name_successes_and_sample_size(self.root_dir + "/corpus/unseen_names.tsv",
                                                                          self.gold_amrs,
                                                                          self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])

        self.set_dataset_name("Seen dates")
        successes, sample_size = calculate_date_successes_and_sample_size(self.root_dir + "/corpus/seen_dates.tsv",
                                                                          self.gold_amrs,
                                                                          self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])

        self.set_dataset_name("Unseen dates")
        successes, sample_size = calculate_date_successes_and_sample_size(self.root_dir + "/corpus/unseen_dates.tsv",
                                                                          self.gold_amrs,
                                                                          self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])

        self.set_dataset_name("Other seen entities")
        successes, sample_size = calculate_special_entity_successes_and_sample_size(self.root_dir + "/corpus/seen_special_entities.tsv",
                                                                                    self.gold_amrs,
                                                                                    self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])

        self.set_dataset_name("Other unseen entities")
        successes, sample_size = calculate_special_entity_successes_and_sample_size(self.root_dir + "/corpus/unseen_special_entities.tsv",
                                                                                    self.gold_amrs,
                                                                                    self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])

    def compute_seen_names_results(self, gold_graphs, predicted_graphs):
        successes, sample_size = calculate_name_successes_and_sample_size(self.root_dir + "/corpus/seen_names.tsv",
                                                                          gold_graphs,
                                                                          predicted_graphs)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]

    def compute_unseen_names_results(self, gold_graphs, predicted_graphs):
        successes, sample_size = calculate_name_successes_and_sample_size(self.root_dir + "/corpus/unseen_names.tsv",
                                                                          gold_graphs,
                                                                          predicted_graphs)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]

    def compute_seen_dates_results(self, gold_graphs, predicted_graphs):
        successes, sample_size = calculate_date_successes_and_sample_size(self.root_dir + "/corpus/seen_dates.tsv",
                                                                          gold_graphs,
                                                                          predicted_graphs)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]

    def compute_unseen_dates_results(self, gold_graphs, predicted_graphs):
        successes, sample_size = calculate_date_successes_and_sample_size(self.root_dir + "/corpus/unseen_dates.tsv",
                                                                          gold_graphs,
                                                                          predicted_graphs)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]

    def compute_seen_special_entities_results(self, gold_graphs, predicted_graphs):
        successes, sample_size = calculate_special_entity_successes_and_sample_size(self.root_dir + "/corpus/seen_special_entities.tsv",
                                                                                    gold_graphs,
                                                                                    predicted_graphs)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]

    def compute_unseen_special_entities_results(self, gold_graphs, predicted_graphs):
        successes, sample_size = calculate_special_entity_successes_and_sample_size(self.root_dir + "/corpus/unseen_special_entities.tsv",
                                                                                    gold_graphs,
                                                                                    predicted_graphs)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]