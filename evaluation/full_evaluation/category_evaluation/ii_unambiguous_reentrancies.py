from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation


class UnambiguousReentrancies(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Syntactic (gap) reentrancies")
        self.rows.extend(self.compute_syntactic_gap_results(self.gold_amrs, self.predicted_amrs))
        self.set_dataset_name("Unambiguous coreference")
        self.rows.extend(self.compute_unambiguous_coreference_results(self.gold_amrs, self.predicted_amrs))

    def compute_syntactic_gap_results(self, gold_amrs, predicted_amrs):
        return self.make_results_columns_for_edge_recall_from_graphs("reentrancies_syntactic_gap_filtered.tsv",
                                                                     gold_amrs, predicted_amrs,
                                                                     parent_column=4,
                                                                     parent_edge_column=5)

    def compute_unambiguous_coreference_results(self, gold_amrs, predicted_amrs):
        return self.make_results_columns_for_edge_recall_from_graphs(
            "reentrancies_unambiguous_coreference_filtered.tsv",
            gold_amrs, predicted_amrs,
            parent_column=4,
            parent_edge_column=5)
