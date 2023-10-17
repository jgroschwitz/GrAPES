from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation


class UnambiguousReentrancies(CategoryEvaluation):

    def _run_all_evaluations(self):
        self.set_dataset_name("Syntactic (gap) reentrancies")
        self.make_results_columns_for_edge_recall("reentrancies_syntactic_gap_filtered.tsv",
                                                  parent_column=4,
                                                  parent_edge_column=5)
        self.set_dataset_name("Unambiguous coreference")
        self.make_results_columns_for_edge_recall("reentrancies_unambiguous_coreference_filtered.tsv",
                                                  parent_column=4,
                                                  parent_edge_column=5)
        # TODO maybe from own grammars

