import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Callable

import penman
from penman import Graph

from evaluation.corpus_metrics import compute_smatch_f_from_graph_lists, graph_is_in_ids
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata, is_sanity_check
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.util import filter_amrs_for_name

EVAL_TYPE_SUCCESS_RATE = "success_rate"
EVAL_TYPE_F1 = "f1"
EVAL_TYPE_NONE = 1
EVAL_TYPE_NA = 0

PREREQS = "prereqs"
UNLABELLED = "unlabelled"

STRUC_GEN = "structural generalisation"


class CategoryEvaluation:

    def __init__(self, gold_amrs: List[Graph], predicted_amrs: List[Graph], category_metadata: SubcategoryMetadata,
                 instance_info: EvaluationInstanceInfo, given_subcorpus_file=False):
        """
        Initialises evaluation of one category
        Args:
            gold_amrs: list of gold penman.Graph
            predicted_amrs: list of one parser's precicted penman.Graph
            category_metadata: SubcategoryMetadata: stores details about evaluating this category,
                                like the files to read in and whether to run prerequisites
        """
        self.gold_amrs = gold_amrs
        self.predicted_amrs = predicted_amrs
        self.root_dir = instance_info.root_dir
        self.corpus_path = f"{self.root_dir}/corpus"
        self.rows = []
        self.category_metadata = category_metadata
        self.print_dataset_name = True  # we want to print the dataset name only on the first metric calculation
        self.extra_subcorpus_filenames = category_metadata.extra_subcorpus_filenames
        self.instance_info = instance_info
        self.is_sanity_check = is_sanity_check(category_metadata)

        # get any extra corpus files needed
        if self.category_metadata.extra_subcorpus_filenames and given_subcorpus_file:
            # if given a subcorpus file rather than the whole corpus, read in new files
            extra_gold, extra_pred = self.get_additional_graphs()
            self.gold_amrs.extend(extra_gold)
            self.predicted_amrs.extend(extra_pred)

        # filter in between because one way to get the extra subcorpus files is by filtering the full corpus files
        self.gold_amrs, self.predicted_amrs = self.filter_graphs()

        # if self.category_metadata.extra_subcorpus_filenames:
        #     self.gold_amrs.extend(extra_gold)
        #     self.predicted_amrs.extend(extra_pred)

        if len(self.predicted_amrs) == 0:
            print("No predicted amrs found!")

        # build empty Results
        extra_fields = self.category_metadata.additional_fields
        if self.category_metadata.run_prerequisites:
            extra_fields.append(PREREQS)
        if self.measure_unlabelled_edges():
            extra_fields.append(UNLABELLED)
        if self.instance_info.do_error_analysis:

            pickle_path = f"{self.instance_info.error_analysis_outdir()}/{self.category_metadata.name}.pickle"
            self.results = IDResults(additional_fields=extra_fields, pickle_path=pickle_path,
                                     verbose=self.instance_info.verbose_error_analysis)
        else:
            self.results = CountResults(additional_fields=extra_fields)

    @staticmethod
    def measure_unlabelled_edges():
        # only true for EdgeRecall
        return False

    def run_evaluation(self):
        """
        Main function.
        Run all evaluations and create output rows
        Returns: results as lists of the form TODO

        """
        self._get_all_results()
        self._calculate_metrics_and_add_all_rows()
        if self.instance_info.run_smatch or self.category_metadata.subtype == STRUC_GEN and not self.is_sanity_check:
            self.make_smatch_results()
        return self.rows

    def get_additional_graphs(self):
        """
        If there are additional graphs required by this category, we can read them in or filter them from the larger set.
        :param: read_in: if True, read them in from a file, otherwise filter them from the stored corpora
        """
        # if not read_in:
        #     filtered_golds = []
        #     filtered_preds = []
        #     for name in self.extra_subcorpus_filenames:
        #         print("Filtering additional subcorpus for", name)
        #         more_golds, more_preds = filter_amrs_for_name(name, self.gold_amrs, self.predicted_amrs)
        #         filtered_golds += more_golds
        #         filtered_preds += more_preds
        #     return filtered_golds, filtered_preds
        # else:
        try:
            extra_predictions = []
            extra_golds = []
            for filename in self.extra_subcorpus_filenames:
                print("reading in", filename)
                extra_predictions += penman.load(f"{self.instance_info.predictions_directory_path()}/{filename}.txt")
                extra_golds += penman.load(f"{self.instance_info.root_dir}/corpus/subcorpora/{filename}.txt")
            return extra_golds, extra_predictions
        except FileNotFoundError as e:
            print(f"Extra files for {self.category_metadata.name} not found in "
                  f"{self.instance_info.predictions_directory_path()}", file=sys.stderr)
            raise e

    def make_and_append_results_row(self, metric_name: str, eval_type: str, metric_results: List):
        """
        :param eval_type: either EVAL_TYPE_SUCCESS_RATE or EVAL_TYPE_F1 (the constants given category_evaluation.py)
        :param metric_name:
        :param metric_results:
        """
        new_row = self.make_results_row(metric_name, eval_type, metric_results)
        self.rows.append(new_row)

    def make_results_row(self, metric_name, eval_type, metric_results):
        """
        Include the main dataset name in the result rows only the first time to reduce clutter.
        Args:
            metric_name: Name of the metric such a Edge Recall
            eval_type: used to guide printing
            metric_results: Output of an evaluation, e.g. [successes, sample_size]
        Returns: new row: [display name if new, metric name, eval type, successes, sample_size]
        """
        if self.print_dataset_name:
            ds_name = self.category_metadata #.display_name TODO
            self.print_dataset_name = False  # don't print it next time
        else:
            ds_name = None  # ""
        new_row = [ds_name, metric_name, eval_type] + metric_results
        return new_row

    @staticmethod
    def make_empty_row(category_name="", metric_name="-"):
        return [category_name, metric_name, EVAL_TYPE_NA] + []

    def make_smatch_results(self):
        smatch = compute_smatch_f_from_graph_lists(self.gold_amrs, self.predicted_amrs)
        smatch_f1 = self.get_f_from_prf(smatch)
        self.make_and_append_results_row("Smatch", EVAL_TYPE_F1, [smatch_f1, len(self.gold_amrs)])

    def get_results_by_size(self):
        """Split up the generalisation by size as marked in corpora.
        Currently just used for structural generalisation"""
        if self.category_metadata.subcorpus_filename in size_mappers:
            return get_exact_match_by_size(self.gold_amrs, self.predicted_amrs,
                                           size_mappers[self.category_metadata.subcorpus_filename])
        else:
            return {}

    @staticmethod
    def get_f_from_prf(triple):
        return triple[2]


    def store_filtered_graphs(self):
        self.gold_amrs, self.predicted_amrs = self.filter_graphs()

    def filter_graphs(self):
        """
        Filter by TSV or id containing info.subcorpus_filename
        If neither works (neither true or we get 0 graphs this way) just return the whole stored graph lists.
        Returns: filtered_golds, filtered_predicted
        """
        filtered_golds = []
        filtered_preds = []
        if self.category_metadata.tsv is not None:
            ids = read_label_tsv(self.root_dir, self.category_metadata.tsv, graph_id_column=self.category_metadata.graph_id_column).keys()
            for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
                if graph_is_in_ids(gold_amr, ids):
                    filtered_golds.append(gold_amr)
                    filtered_preds.append(predicted_amr)
        elif self.category_metadata.subcorpus_filename is not None:
            filtered_golds, filtered_preds = filter_amrs_for_name(self.category_metadata.subcorpus_filename, self.gold_amrs, self.predicted_amrs, fail_ok=True)
            if self.category_metadata.extra_subcorpus_filenames is not None:
                for name in self.category_metadata.extra_subcorpus_filenames:
                    more_gold, more_preds = filter_amrs_for_name(name, self.gold_amrs, self.predicted_amrs,
                                                                          fail_ok=False)
                    filtered_golds.extend(more_gold)
                    filtered_preds.extend(more_preds)
        else:
            print("No filtering done")
            filtered_golds = self.gold_amrs
            filtered_preds = self.predicted_amrs
        if len(filtered_golds) == 0:
            print("WARNING: filtering gave us 0 graphs!", file=sys.stderr)
            return [],[]
        return filtered_golds, filtered_preds

    def get_all_gold_ids(self):
        filtered_gold, _ = self.filter_graphs()
        return [gold.metadata["id"] for gold in filtered_gold]

    def read_tsv(self):
        return read_label_tsv(self.root_dir, self.category_metadata.tsv)

    # def dump_error_analysis_pickle(self):
    #     if self. do_error_analysis:
    #         try:
    #             self.results.write_pickle()
    #         except Exception as e:
    #             print("WARNING: no error analysis written:", e, file=sys.stderr)
    #     else:
    #         print("No error analysis data was stored, so no pickle to write")

    def _calculate_metrics_and_add_all_rows(self):

        success_count = self.get_success_count()
        sample_size = success_count + self.get_failure_count()
        assert sample_size > 0, "No results for _calculate_metrics_and_add_all_rows"
        ret = [success_count, sample_size]

        self.rows.append(self.make_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE,
                                               [success_count, sample_size]))
        if self.measure_unlabelled_edges():
            unlabelled_success_count = self.get_success_count(UNLABELLED)
            ret.append(unlabelled_success_count)
            self.rows.append(self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE,
            [unlabelled_success_count, sample_size]))
        if self.category_metadata.run_prerequisites:
            prereq_success_count = self.get_success_count(PREREQS)
            ret.append(prereq_success_count)
            self.rows.append(self.make_results_row(
                "Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq_success_count, sample_size]))

        if self.instance_info.do_error_analysis:
            self.results.write_pickle()
        # print("Metrics:", ret)
        return ret

    def get_predictions_for_comparison(self, predicted_amr):
        """
        implement for a handy way to get something other than just hte predicted amr for comparison to the target
        """
        return None

    def _get_all_results(self):
        """
        Loops through graphs and updates results for each instance to evaluate
        """
        if self.category_metadata.tsv is not None:
            # read in the TSV to get the targets for comparison
            id2labels = self.read_tsv()
            for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
                graph_id = gold_amr.metadata['id']
                if graph_id in id2labels:
                    # occasionally we need something other than just the predicted graph for comparison
                    predictions_for_comparison = self.get_predictions_for_comparison(predicted_amr)
                    for target in id2labels[graph_id]:
                        # update results for this item
                        # if we have a TSV, update_results is per item in the TSV
                        self.update_results(gold_amr, predicted_amr, target, predictions_for_comparison)
        else:
            self.gold_amrs, self.predicted_amrs = self.filter_graphs()
            assert len(self.gold_amrs) > 0, "No matching AMRs in given corpus"
            for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
                # if no TSV, update_results is per graph pair
                self.update_results(gold_amr, predicted_amr, None, None)

    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison):
        """
        Default: exact match, modulo edge labels and senses
        Args:
            gold_amr: Graph
            predicted_amr: Graph
            target: optional: gold thing to match with.
            predictions_for_comparison: optional: predicted thing to match.
        """
        if equals_modulo_isomorphy(gold_amr, predicted_amr, match_edge_labels=False, match_senses=False):
            self.add_success(gold_amr, predicted_amr)
        else:
            self.add_fail(gold_amr, predicted_amr)

    def add_success(self, gold: Graph, predicted, field=None):
        self.results.add_success(gold, predicted, field)

    def add_fail(self, gold, predicted, field=None):
        self.results.add_fail(gold, predicted, field)

    def get_success_count(self, field=None):
        return self.results.get_success_count(field)

    def get_failure_count(self, field=None):
        return self.results.get_fail_count(field)


class Results(ABC):
    """
    Stores the results of a category evaluation.
    Can be implemented to, for example, create a pickled corpus of correct and incorrect graphs.
    """
    def __init__(self, additional_fields: List[str] = None, default_field: str = "id", verbose=True):
        """
        For each field, including the default, initialise a way to store results
        For example, CountResults initialises variables: self.correct_ids = 0, self.incorrect_ids = 0, etc
         while IDResults initialises a dict with these as keys: {"correct_ids" = [], "incorrect_ids"=[]}, etc.
        These variables/keys can be made with self.make_success_key and self.make_failure_key.
        Args:
            verbose:
            additional_fields: list of names of types of results to store. e.g. "prereq"
            default_field: for the main analysis.
        """
        self.default_field = default_field
        self.verbose = verbose

    @abstractmethod
    def add_success(self, gold: Graph, predicted: Graph, field:str=None):
        """
        Updates the success counter by one
            For example, CountResults just increments the count,
            and IDResults adds the graph ID to a dictionary under "correct_<field>".
            Both graphs are provided as arguments in case you want to do something like store them in a Vulcan-readable pickle.
        Args:
            gold: gold AMR
            predicted: predicted AMR
            field: name for the kind of result we're storing. Default is usually just "id" as in graph id,
             but common alternatives are "prereq" and "unlabelled"
        """
        raise NotImplementedError

    @abstractmethod
    def add_fail(self, gold: Graph, predicted: Graph, field:str=None):
        """
        Updates the failure counter by one
            For example, CountResults just increments the count,
            and IDResults adds the graph ID to a dictionary under "correct_<field>".
            Both graphs are provided as arguments in case you want to do something like store them in a Vulcan-readable pickle.
        Args:
            gold: gold AMR
            predicted: predicted AMR
            field: name for the kind of result we're storing. Default is usually just "id" as in graph id,
             but common alternatives are "prereq" and "unlabelled"
        """
        raise NotImplementedError

    @abstractmethod
    def get_success_count(self, field=None):
        """
        Use the stored successes to get their actual count.
            for example, CountResults just gets the stored count, while IDResults gets the length of the stored IDs.
        Args:
            field: name for the kind of result we're storing. Default is usually just "id" as in graph id,
             but common alternatives are "prereq" and "unlabelled"
        Returns: int
        """
        raise NotImplementedError

    @abstractmethod
    def get_fail_count(self, field=None):
        """
        Use the stored successes to get their actual count.
            for example, CountResults just gets the stored count, while IDResults gets the length of the stored IDs.
        Args:
            field: name for the kind of result we're storing. Default is usually just "id" as in graph id,
             but common alternatives are "prereq" and "unlabelled"
        Returns: int
        """
        raise NotImplementedError

    def make_success_key(self, field=None):
        """Default way to name correct results"""
        if field is None:
            field = self.default_field
        return f"correct_{field}"

    def make_fail_key(self, field=None):
        """Default way to name incorrect results"""
        if field is None:
            field = self.default_field
        return f"incorrect_{field}"



class CountResults(Results):
    """
    Just counts everything, no storage of which graph is which
    """
    def __init__(self, additional_fields: List[str]=None, default_field: str = "id"):
        super().__init__(additional_fields, default_field)
        if additional_fields is None:
            additional_fields = []
        for field in [default_field] + additional_fields:
            setattr(self, self.make_success_key(field), 0)
            setattr(self, self.make_fail_key(field), 0)

    def add_success(self, gold: Graph, predicted: Graph, success_type=None):
        if success_type is None:
            success_type = self.default_field
        key = self.make_success_key(success_type)
        setattr(self, key, getattr(self, key) + 1)

    def add_fail(self, gold: Graph, predicted: Graph, failure_type=None):
        if failure_type is None:
            failure_type = self.default_field
        key = self.make_fail_key(failure_type)
        setattr(self, key, getattr(self, key) + 1)

    def get_success_count(self, field=None):
        if field is None:
            field = self.default_field
        return getattr(self, self.make_success_key(field))

    def get_fail_count(self, field=None):
        if field is None:
            field = self.default_field
        return getattr(self, self.make_fail_key(field))

class IDResults(Results):
    """
    Stores graph IDs of predictions that did and did not pass the evaluation.
    """
    def __init__(self, additional_fields: List[str]=None, default_field: str = "id", pickle_path: str=None,
                 verbose=True):
        super().__init__(additional_fields, default_field, verbose=verbose)

        self.error_analysis_dict = {}
        self.pickle_path = pickle_path

        if additional_fields is None:
            additional_fields = []
        for field in [default_field] + additional_fields:
            self.error_analysis_dict[self.make_success_key(field)] = []
            self.error_analysis_dict[self.make_fail_key(field)] = []

    def add_success(self, gold: Graph, predicted: Graph, field=None):
        self.error_analysis_dict[self.make_success_key(field)].append(gold.metadata["id"])

    def add_fail(self, gold: Graph, predicted: Graph, field=None):
        self.error_analysis_dict[self.make_fail_key(field)].append(gold.metadata["id"])

    def get_success_count(self, field=None):
        return len(self.error_analysis_dict[self.make_success_key(field)])

    def get_fail_count(self, field=None):
        return len(self.error_analysis_dict[self.make_fail_key(field)])

    def write_pickle(self):
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.error_analysis_dict, f)
        if self.verbose:
            print("Wrote error analysis pickle to " + self.pickle_path)




def get_exact_match_by_size(gold_graphs: List[Graph], predicted_graphs: List[Graph],
                            size_mapper: Callable[[int], int] = lambda x: x):
    assert len(gold_graphs) == len(predicted_graphs)
    correct_counts = Counter()
    total_counts = Counter()
    for gold, prediction in zip(gold_graphs, predicted_graphs):
        try:
            size = size_mapper(int(gold.metadata["size0"]))
        except KeyError:
            print(f"No size0 found in {gold.metadata.keys()}!")
            print(gold.metadata["id"])
            raise
        total_counts[size] += 1
        if equals_modulo_isomorphy(gold, prediction, match_edge_labels=False, match_senses=False):
            correct_counts[size] += 1
    ret = {size: correct_counts[size] / total_counts[size] for size in sorted(total_counts.keys())}
    ret["total"] = sum(correct_counts.values()) / sum(total_counts.values())
    return ret

size_mappers = {"adjectives": lambda x: x - 2,
                "centre_embedding": lambda x: (x - 2) // 2,
                "nested_control": lambda x: x,
                "deep_recursion_basic": lambda x: x - 1,
                "deep_recursion_pronouns": lambda x: x - 1,
                "deep_recursion_3s": lambda x: x - 1,
                "deep_recursion_rc": lambda x: x + 1,
                "deep_recursion_rc_contrastive_coref": lambda x: x + 1}