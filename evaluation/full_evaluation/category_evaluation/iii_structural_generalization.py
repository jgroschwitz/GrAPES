from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1
from evaluation.structural_generalization import get_all_success_counts
from evaluation.long_lists import evaluate_long_lists, evaluate_long_lists_generalization, compute_conjunct_counts, \
    compute_generalization_op_counts
from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size, compute_smatch_f, \
    compute_smatch_f_from_graph_lists

from penman import load


class StructuralGeneralization(CategoryEvaluation):

    def _run_all_evaluations(self):
        # run the evaluation on all
        struct_gen_results = get_all_success_counts(self.parser_name, root_dir=self.root_dir)

        # make all the tables
        self.set_dataset_name("Nested control and coordination")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["nested_control"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["nested_control"][2])])

        self.set_dataset_name("Sanity check")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["nested_control_sanity_check"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["nested_control_sanity_check"][2])])

        self.set_dataset_name("Multiple adjectives")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE, struct_gen_results["adjectives"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["adjectives"][2])])

        self.set_dataset_name("Sanity check")
        self.make_and_append_results_row("Exact match",
                                         EVAL_TYPE_SUCCESS_RATE, struct_gen_results["adjectives_sanity_check"][0:2])
        self.make_and_append_results_row("Smatch",
                                         EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["adjectives_sanity_check"][2])])

        self.set_dataset_name("Centre embedding")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["centre_embedding"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["centre_embedding"][2])])

        self.set_dataset_name("Sanity check")
        self.make_and_append_results_row("Exact match",
                                         EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["centre_embedding_sanity_check"][0:2])
        self.make_and_append_results_row("Smatch",
                                         EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["centre_embedding_sanity_check"][2])])

        self.set_dataset_name("CP recursion")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["deep_recursion_basic"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["deep_recursion_basic"][2])])

        self.set_dataset_name("Sanity check")
        self.make_and_append_results_row("Exact match",
                                         EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["deep_recursion_basic_sanity_check"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(
                                             struct_gen_results["deep_recursion_basic_sanity_check"][2])])

        deep_recursion_with_coref_results = [struct_gen_results["deep_recursion_pronouns"][0]
                                             + struct_gen_results["deep_recursion_3s"][0],
                                             struct_gen_results["deep_recursion_pronouns"][1]
                                             + struct_gen_results["deep_recursion_3s"][1],
                                             (self.get_f_from_prf(struct_gen_results["deep_recursion_pronouns"][2])
                                              + self.get_f_from_prf(struct_gen_results["deep_recursion_3s"][
                                                                        2])) / 2]  # taking the average for smatch (not exactly correct, since this overvalues the larger corpus, but the sizes should be close enough. # we get a triple of p, r, f and want the average of the two f scores
        deep_recursion_with_coref_results_sanity_check = [struct_gen_results["deep_recursion_pronouns_sanity_check"][0]
                                                          + struct_gen_results["deep_recursion_3s_sanity_check"][0],
                                                          struct_gen_results["deep_recursion_pronouns_sanity_check"][1]
                                                          + struct_gen_results["deep_recursion_3s_sanity_check"][1],
                                                          (self.get_f_from_prf(
                                                              struct_gen_results[
                                                                  "deep_recursion_pronouns_sanity_check"][2])
                                                           + self.get_f_from_prf(struct_gen_results[
                                                                                     "deep_recursion_3s_sanity_check"][
                                                                                     2]
                                                                                 )) / 2]  # we get a triple of p, r, f and want the average of the two f scores

        self.set_dataset_name("CP recursion + coreference")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE, deep_recursion_with_coref_results[0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1, [deep_recursion_with_coref_results[2]])

        self.set_dataset_name("Sanity check")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         deep_recursion_with_coref_results_sanity_check[0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [deep_recursion_with_coref_results_sanity_check[2]])

        self.set_dataset_name("CP recursion + relative clause (RC)")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["deep_recursion_rc"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["deep_recursion_rc"][2])])

        self.set_dataset_name("Sanity check")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["deep_recursion_rc_sanity_check"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(struct_gen_results["deep_recursion_rc_sanity_check"][2])])

        self.set_dataset_name("CP recursion + RC + coreference")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["deep_recursion_rc_contrastive_coref"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1,
                                         [self.get_f_from_prf(
                                             struct_gen_results["deep_recursion_rc_contrastive_coref"][2])])

        self.set_dataset_name("Sanity check")
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         struct_gen_results["deep_recursion_rc_contrastive_coref_sanity_check"][0:2])
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1, [
            self.get_f_from_prf(struct_gen_results["deep_recursion_rc_contrastive_coref_sanity_check"][2])])

        golds_long_lists, golds_long_lists_sanity_check, \
        predictions_long_lists, predictions_long_lists_sanity_check = self.load_long_list_amrs()

        # op_f1, conjunct_f1 = evaluate_long_lists(predictions=predictions_long_lists, golds=golds_long_lists)
        # generalization_op_f1, generalization_conjunct_f1 = evaluate_long_lists_generalization(
        #     predictions=predictions_long_lists, golds=golds_long_lists
        # )
        self.set_dataset_name("Long lists")

        conj_total_gold, conj_total_predictions, conj_true_predictions = compute_conjunct_counts(golds_long_lists,
                                                                                                 predictions_long_lists)

        self.make_and_append_results_row("Conjunct recall", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_gold])
        self.make_and_append_results_row("Conjunct precision", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_predictions])
        print("conj total predictions: ", conj_total_predictions)

        opi_gen_total_gold, opi_gen_total_predictions, opi_gen_true_predictions = compute_generalization_op_counts(
            golds_long_lists, predictions_long_lists
        )
        self.make_and_append_results_row("Unseen :opi recall", EVAL_TYPE_SUCCESS_RATE,
                                         [opi_gen_true_predictions, opi_gen_total_gold])

        # self.make_results_column(":opi f1", EVAL_TYPE_F1, [self.get_f_from_prf(op_f1)])
        # self.make_results_column("Conjunct f1", EVAL_TYPE_F1, [self.get_f_from_prf(conjunct_f1)])
        # self.make_results_column("Unseen :opi f1", EVAL_TYPE_F1, [self.get_f_from_prf(generalization_op_f1)])
        # self.make_results_column("Conjunct f1 for unseen :opi", EVAL_TYPE_F1,
        #                          [self.get_f_from_prf(generalization_conjunct_f1)])

        self.set_dataset_name("Sanity check")
        success, sample_size = compute_exact_match_successes_and_sample_size(golds_long_lists_sanity_check,
                                                                             predictions_long_lists_sanity_check,
                                                                             match_edge_labels=False,
                                                                             match_senses=False)
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE, [success, sample_size])

    def compute_exact_match_and_f1_results(self, gold_graphs, predicted_graphs):
        """
        This function can be used for all categories here, except for long lists
        """
        successes, sample_size = compute_exact_match_successes_and_sample_size(gold_graphs, predicted_graphs,
                                                                               match_edge_labels=False,
                                                                               match_senses=False)

        smatch_f1 = compute_smatch_f_from_graph_lists(gold_graphs, predicted_graphs)
        return [self.make_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size]),
                self.make_results_row("Smatch", EVAL_TYPE_F1, [smatch_f1])]

    def compute_long_list_results(self, gold_graphs, predicted_graphs):
        """
        This function can be used for long lists proper. For long list sanity checks,
         use compute_exact_match_and_f1_results
        """
        conj_total_gold, conj_total_predictions, conj_true_predictions = compute_conjunct_counts(gold_graphs,
                                                                                                 predicted_graphs)
        ret = []
        ret.append(self.make_results_row("Conjunct recall", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_gold]))
        ret.append(self.make_results_row("Conjunct precision", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_predictions]))
        print("conj total predictions: ", conj_total_predictions)

        opi_gen_total_gold, opi_gen_total_predictions, opi_gen_true_predictions = compute_generalization_op_counts(
            gold_graphs, predicted_graphs
        )
        ret.append(self.make_results_row("Unseen :opi recall", EVAL_TYPE_SUCCESS_RATE,
                                         [opi_gen_true_predictions, opi_gen_total_gold]))
        return ret

    def load_long_list_amrs(self):
        gold_file_path_long_lists = f"{self.root_dir}/corpus/long_lists.txt"
        golds_long_lists = load(gold_file_path_long_lists)
        prediction_folder = self.root_dir + "/" + self.parser_name + "-output/"
        prediction_file_path_long_lists = prediction_folder + "long_lists.txt"
        predictions_long_lists = load(prediction_file_path_long_lists)
        gold_file_path_long_lists_sanity_check = f"{self.root_dir}/corpus/long_lists_sanity_check.txt"
        golds_long_lists_sanity_check = load(gold_file_path_long_lists_sanity_check)
        prediction_file_path_long_lists_sanity_check = prediction_folder + "long_lists_sanity_check.txt"
        predictions_long_lists_sanity_check = load(prediction_file_path_long_lists_sanity_check)
        return golds_long_lists, golds_long_lists_sanity_check, predictions_long_lists, predictions_long_lists_sanity_check
