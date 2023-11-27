from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1

from evaluation.pp_attachment import get_pp_attachment_success_counters

from evaluation.testset.ne_types import get_ne_type_successes_and_sample_size

class Attachments(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("PP attachment")
        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.root_dir+"/corpus/",
                                                                               self.root_dir+f"/{self.parser_name}-output/")
        self.make_and_append_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])

        # TODO we may want to break this down by category (and maybe also distance) in the more finegrained evaluation
        # category is in column 6; distance is in 5
        self.set_dataset_name("Unbounded dependencies")
        ccg_golds, ccg_preds = self.get_gold_and_pred_for_corpus("unbounded_dependencies")
        self.make_results_columns_for_edge_recall("unbounded_dependencies.tsv", use_sense=False,
                                                  override_gold_amrs=ccg_golds,
                                                  override_predicted_amrs=ccg_preds,
                                                  source_column=2, edge_column=3, target_column=4,
                                                  first_row_is_header=True)

        self.set_dataset_name("Passives")
        self.make_results_columns_for_edge_recall("passives_filtered.tsv", use_sense=True)

        self.set_dataset_name("Unaccusatives")
        self.make_results_columns_for_edge_recall("unaccusatives2_filtered.tsv", use_sense=True)


