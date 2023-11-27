from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation


class PragmaticReentrancies(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Pragmatic coreference (testset)")
        self.rows.extend(self.compute_testset_results(self.gold_amrs, self.predicted_amrs))

        self.set_dataset_name("Pragmatic coreference (Winograd)")
        winograd_gold, winograd_pred = self.get_gold_and_pred_for_corpus("winograd")
        self.rows.extend(self.compute_winograd_results(winograd_gold, winograd_pred))

    def compute_testset_results(self, gold_amrs, predicted_amrs):
        return self.make_results_columns_for_edge_recall_from_graphs("reentrancies_pragmatic_filtered.tsv",
                                                                     gold_amrs, predicted_amrs,
                                                                     parent_column=4,
                                                                     parent_edge_column=5)

    def compute_winograd_results(self, gold_amrs, predicted_amrs):
        return self.make_results_columns_for_edge_recall_from_graphs("winograd_annotated.tsv",
                                                                     gold_amrs, predicted_amrs,
                                                                     parent_column=4,
                                                                     parent_edge_column=5,
                                                                     first_row_is_header=True)
