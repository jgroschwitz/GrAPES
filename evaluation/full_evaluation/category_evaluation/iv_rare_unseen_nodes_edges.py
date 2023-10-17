from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation
from penman import load


class RareUnseenNodesEdges(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Rare node labels")
        self.make_results_column_for_node_recall("rare_node_labels_test.tsv", use_sense=True)

        self.set_dataset_name("Unseen node labels")
        self.make_results_column_for_node_recall("unseen_node_labels_test_filtered.tsv", use_sense=True)

        self.set_dataset_name("Rare predicate senses (excl.~\\nl{-01})")
        self.make_results_column_for_node_recall("rare_senses_filtered.tsv", use_sense=True)
        self.make_results_column_for_node_recall("rare_senses_filtered.tsv", use_sense=False,
                                                 metric_label="Prerequisites")

        unseen_senses_own_gold, unseen_senses_own_pred = self.get_gold_and_pred_for_corpus(
            "unseen_senses_new_sentences")
        self.set_dataset_name("Unseen predicate senses (excl~\\nl{-01})")
        self.make_results_column_for_node_recall("unseen_senses_new_sentences.tsv", use_sense=True,
                                                 override_gold_amrs=unseen_senses_own_gold,
                                                 override_predicted_amrs=unseen_senses_own_pred)
        self.make_results_column_for_node_recall("unseen_senses_new_sentences.tsv", use_sense=False,
                                                 metric_label="Prerequisites",
                                                 override_gold_amrs=unseen_senses_own_gold,
                                                 override_predicted_amrs=unseen_senses_own_pred)

        self.set_dataset_name("Rare edge labels (\\nl{ARG2}+)")
        
        self.make_results_columns_for_edge_recall("rare_roles_arg2plus_filtered.tsv", use_sense=True)

        unseen_roles_own_gold, unseen_roles_own_pred = self.get_gold_and_pred_for_corpus("unseen_roles_new_sentences")
        self.set_dataset_name("Unseen edge labels (\\nl{ARG2}+)")
        self.make_results_columns_for_edge_recall("unseen_roles_new_sentences.tsv", use_sense=True,
                                                  override_gold_amrs=unseen_roles_own_gold,
                                                  override_predicted_amrs=unseen_roles_own_pred)
