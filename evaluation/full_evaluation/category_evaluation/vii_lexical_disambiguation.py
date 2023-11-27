from evaluation import util
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1

from evaluation.testset.ne_types import get_ne_type_successes_and_sample_size
from evaluation.berts_mouth import evaluate_berts_mouth
from evaluation.word_disambiguation import evaluate_word_disambiguation

class LexicalDisambiguation(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Frequent predicate senses (incl -01)")
        self.make_results_column_for_node_recall("common_senses_filtered.tsv", use_sense=True)
        self.make_results_column_for_node_recall("common_senses_filtered.tsv", use_sense=False,
                                                 metric_label="Prerequisites")

        word_disambiguation_gold, word_disambiguation_pred = self.get_gold_and_pred_for_corpus("word_disambiguation")
        self.set_dataset_name("Word ambiguities (handcrafted)")
        successes, sample_size = evaluate_word_disambiguation(word_disambiguation_gold, word_disambiguation_pred)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])


        # TODO fix have-06 usage for eating and drinking (after talking to lucia if there are updated annotations coming)
        berts_mouth_gold, berts_mouth_pred = self.get_gold_and_pred_for_corpus("berts_mouth")
        self.set_dataset_name("Word ambiguities \cite{karidi-etal-2021-putting}")
        successes, sample_size = evaluate_berts_mouth(berts_mouth_gold, berts_mouth_pred)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])

    def compute_common_senses_results(self, gold_graphs, predicted_graphs):
        return [self.make_results_column_for_node_recall_from_graphs("common_senses_filtered.tsv",
                                                                     gold_graphs,
                                                                     predicted_graphs,
                                                                     use_sense=True),
                self.make_results_column_for_node_recall_from_graphs("common_senses_filtered.tsv",
                                                                     gold_graphs,
                                                                     predicted_graphs,
                                                                     use_sense=False,
                                                                     metric_label="Prerequisites")]

    def compute_grapes_word_disambiguation_results(self, gold_graphs, predicted_graphs):
        successes, sample_size = evaluate_word_disambiguation(gold_graphs, predicted_graphs)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]

    def compute_berts_mouth_results(self, gold_graphs, predicted_graphs):
        filtered_gold, filtered_pred = util.filter_amrs_for_name("berts_mouth", gold_graphs, predicted_graphs)
        successes, sample_size = evaluate_berts_mouth(filtered_gold, filtered_pred)
        return [self.make_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])]

