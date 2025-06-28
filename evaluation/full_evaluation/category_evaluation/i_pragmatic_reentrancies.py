from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation
from subcategory_info import category_name_to_subcategory_info

class PragmaticReentrancies(CategoryEvaluation):

    def _run_all_evaluations(self):
        for category in ["pragmatic_coreference_testset", "pragmatic_coreference_winograd"]:
            subcategory_data = category_name_to_subcategory_info[category]
            self.set_dataset_name(subcategory_data.display_name)
            results = self.make_results_columns_for_edge_recall_from_graphs(
                subcategory_data,
                self.gold_amrs, self.predicted_amrs,
                )
            self.rows.extend(results)

        # self.set_dataset_name("Pragmatic coreference (testset)")
        # self.rows.extend(self.compute_testset_results(self.gold_amrs, self.predicted_amrs))
        #
        # self.set_dataset_name("Pragmatic coreference (Winograd)")
        # winograd_gold, winograd_pred = self.get_gold_and_pred_for_corpus("winograd")
        # self.rows.extend(self.compute_winograd_results(winograd_gold, winograd_pred))

    def compute_testset_results(self, gold_amrs, predicted_amrs):
        return self.make_results_columns_for_edge_recall_from_graphs(category_name_to_subcategory_info["reentrancies_pragmatic_filtered.tsv"],
                                                                     gold_amrs, predicted_amrs,
                                                                     )

    def compute_winograd_results(self, gold_amrs, predicted_amrs):
        return self.make_results_columns_for_edge_recall_from_graphs("winograd.tsv",
                                                                     gold_amrs, predicted_amrs,
                                                                     parent_column=4,
                                                                     parent_edge_column=5,
                                                                     first_row_is_header=True)
