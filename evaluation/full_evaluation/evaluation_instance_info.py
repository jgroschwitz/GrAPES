import dataclasses
import os


@dataclasses.dataclass
class EvaluationInstanceInfo:
    root_dir: str = "."
    run_smatch: bool = False

    # paths
    path_to_grapes_predictions_file_from_root: str = None
    subcorpus_predictions_directory_path_from_root: str = None
    full_pred_grapes_name: str = "full_corpus"
    full_pred_testset_name: str = "testset"
    gold_testset_name: str = "test"
    gold_testset_directory_path_from_root: str = "data/raw/gold"
    # in case it's in a different directory from the subcorpora
    path_to_full_testset_predictions_file_from_root: str = None
    path_to_gold_testset_file_from_root: str = None
    result_output_parent_path_from_root: str = "data/processed/results"

    given_single_file=False

    # for error analysis
    do_error_analysis: bool = False
    verbose_error_analysis: bool = True
    parser_name: str = "unnamed_parser"
    # error_analysis_outdir_from_root: str = f"error_analysis/{parser_name}"

    # for the script running the evaluation
    fail_ok: int = 0

    # for table writing
    # in the paper we don't include Smatch and unlabelled edges
    print_f1_default: bool = False
    print_unlabeled_edge_attachment: bool = False

    def print_f1(self):
        return self.print_f1_default or self.run_smatch

    # path functions
    def gold_testset_path(self):
        if self.path_to_gold_testset_file_from_root is None:
            if self.gold_testset_directory_path_from_root is not None:
                return f"{self.root_dir}/{self.gold_testset_directory_path_from_root}/{self.gold_testset_name}.txt"
            else:
                return None
        else:
            return f"{self.root_dir}/{self.path_to_gold_testset_file_from_root}"

    def predictions_directory_path(self):
        if self.subcorpus_predictions_directory_path_from_root is None:
            if self.path_to_grapes_predictions_file_from_root is not None:
                return os.path.dirname(self.path_to_grapes_predictions_file_from_root)
            else:
                return None
        else:
            return f"{self.root_dir}/{self.subcorpus_predictions_directory_path_from_root}"

    def full_grapes_pred_file_path(self):
        if self.path_to_grapes_predictions_file_from_root is None:
            if self.subcorpus_predictions_directory_path_from_root is not None:
                return f"{self.root_dir}/{self.subcorpus_predictions_directory_path_from_root}/{self.full_pred_grapes_name}.txt"
            else:
                return None
        else:
            return f"{self.root_dir}/{self.path_to_grapes_predictions_file_from_root}"

    def results_directory_path(self):
        parent = f"{self.root_dir}/{self.result_output_parent_path_from_root}"
        return f"{parent}/{self.parser_name}"

    def error_analysis_outdir(self):
        return f"{self.root_dir}/error_analysis/{self.parser_name}"

    def testset_pred_file_path(self):
        """default full_corpus.txt in same directory as subcorpora"""
        if self.path_to_full_testset_predictions_file_from_root is not None:
            return f"{self.root_dir}/{self.path_to_full_testset_predictions_file_from_root}"
        else:
            return None

    def default_testset_pred_file_path(self):
        predpath = self.predictions_directory_path()
        if predpath is None:
            return None
        else:
            return f"{predpath}/{self.full_pred_testset_name}.txt"

    def gold_grapes_path(self):
        return f"{self.root_dir}/corpus/corpus.txt"