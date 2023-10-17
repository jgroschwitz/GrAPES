from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation


class PragmaticReentrancies(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Pragmatic coreference (testset)")
        self.make_results_columns_for_edge_recall("reentrancies_pragmatic_filtered.tsv",
                                                  parent_column=4,
                                                  parent_edge_column=5)

        self.set_dataset_name("Pragmatic coreference (Winograd)")
        winograd_gold, winograd_pred = self.get_gold_and_pred_for_corpus("winograd")
        self.make_results_columns_for_edge_recall("winograd_annotated.tsv",
                                                  parent_column=4,
                                                  parent_edge_column=5,
                                                  override_gold_amrs=winograd_gold,
                                                  override_predicted_amrs=winograd_pred,
                                                  first_row_is_header=True)

