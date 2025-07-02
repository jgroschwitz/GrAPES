from evaluation.corpus_metrics import _check_prerequisites_for_edge_tuple, check_edge_existence, \
    _check_edge_existence_with_multiple_label_options
from evaluation.file_utils import read_edge_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, PREREQS, UNLABELLED


class EdgeRecall(CategoryEvaluation):
    def run_evaluation(self):
        try:
            self._get_all_results()
            self._calculate_metrics_and_add_all_rows()
            return self.rows
        except IndexError as e:
            if self.category_metadata.subcorpus_filename == "unbounded_dependencies":
                print("Check that corpus/unbounded_dependencies.tsv has 66 rows")
                print("Something may have gone wrong in extending the GrAPES testset with PTB data")
            raise e
        except FileNotFoundError as e:
            if self.category_metadata.subcorpus_filename == "unbounded_dependencies":
                print("Check that corpus/unbounded_dependencies.tsv exists")
                print("Something may have gone wrong in extending the GrAPES testset with PTB data")
            raise e

    @staticmethod
    def measure_unlabelled_edges():
        return True

    def read_tsv(self):
        return read_edge_tsv(self.root_dir, self.category_metadata)

    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):
        prereqs_ok = _check_prerequisites_for_edge_tuple(target, predicted_amr)
        if prereqs_ok:
            self.add_success(gold_amr, predicted_amr, PREREQS)
            unlabeled_edge_found = check_edge_existence(target, predicted_amr,
                                                        match_edge_labels=False,
                                                        match_senses=self.category_metadata.use_sense)
            if unlabeled_edge_found:
                self.add_success(gold_amr, predicted_amr, UNLABELLED)
                edge_found = _check_edge_existence_with_multiple_label_options(target, predicted_amr,
                                                      use_sense=self.category_metadata.use_sense)
                if edge_found:
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)
            else:
                self.add_fail(gold_amr, predicted_amr, UNLABELLED)
                self.add_fail(gold_amr, predicted_amr)
        else:
            self.add_fail(gold_amr, predicted_amr, PREREQS)
            self.add_fail(gold_amr, predicted_amr, UNLABELLED)
            self.add_fail(gold_amr, predicted_amr)
