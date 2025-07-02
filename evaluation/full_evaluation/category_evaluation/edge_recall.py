from evaluation.corpus_metrics import _check_prerequisites_for_edge_tuple, check_edge_existence, \
    _check_edge_existence_with_multiple_label_options
from evaluation.file_utils import read_edge_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation


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

    def update_error_analysis(self, graph_id, predicted, target):
        prereqs_ok = _check_prerequisites_for_edge_tuple(target, predicted)
        if prereqs_ok:
            self.add_prereq_success(graph_id)
            unlabeled_edge_found = check_edge_existence(target, predicted,
                                                        match_edge_labels=False,
                                                        match_senses=self.category_metadata.use_sense)
            if unlabeled_edge_found:
                self.add_unlabelled_success(graph_id)
                edge_found = _check_edge_existence_with_multiple_label_options(target, predicted,
                                                      use_sense=self.category_metadata.use_sense)
                if edge_found:
                    self.add_success(graph_id)
                else:
                    self.add_fail(graph_id)
            else:
                self.add_unlabelled_fail(graph_id)
                self.add_fail(graph_id)
        else:
            self.add_prereq_fail(graph_id)
            self.add_unlabelled_fail(graph_id)
            self.add_fail(graph_id)
