import pickle

from amrbank_analysis.get_unseen_names_and_dates import get_date_string_for_date_instance, \
    get_name_string_for_name_instance
from evaluation.novel_corpus.berts_mouth import evaluate_berts_mouth
from evaluation.corpus_metrics import compute_exact_match_successes_and_sample_size, \
    calculate_subgraph_existence_successes_and_sample_size, \
    calculate_node_label_successes_and_sample_size, calculate_edge_prereq_recall_and_sample_size_counts, \
    graph_is_in_ids, _get_predicted_labels_based_on_evaluation_case, _label_exists_in_predicted_labels, \
    _check_prerequisites_for_edge_tuple, check_edge_existence, _check_edge_existence_with_multiple_label_options
from evaluation.file_utils import read_label_tsv, read_edge_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata, is_sanity_check
from evaluation.novel_corpus.long_lists import compute_conjunct_counts, compute_generalization_op_counts
from evaluation.novel_corpus.pp_attachment import get_pp_attachment_success_counters
from evaluation.novel_corpus.structural_generalization import get_exact_match_by_size, size_mappers, \
    add_sanity_check_suffix
from evaluation.testset.ellipsis import get_ellipsis_success_counts
from penman import load

from evaluation.testset.imperative import get_imperative_success_counts
from evaluation.testset.ne_types import get_2_columns_from_tsv_by_id, get_ne_type_successes_and_sample_size
from evaluation.testset.special_entities import get_graphid2labels_from_tsv_file, \
    calculate_special_entity_successes_and_sample_size, normalize_special_entity_value
from evaluation.util import filter_amrs_for_name, get_node_name_for_gold_label, strip_sense, get_node_by_name, get_name
from evaluation.novel_corpus.word_disambiguation import evaluate_word_disambiguation


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


class NodeRecall(CategoryEvaluation):

    def _get_predicted_labels_based_on_evaluation_case(self, predicted_amr, use_sense=None):
        """
        Get the instances or attributes in the given predicted AMR
        Note that if use_attributes and use_sense are both true, we get the attributes, not the senses.
            If they are both false, we get the instances without their senses.
        :param predicted_amr: AMR to search through
        :param use_sense: if True, get all instances with their senses; otherwise all instances without their senses
        :return: list of either attributes or senses (not both)
        """
        if use_sense is None:
            use_sense = self.category_metadata.use_sense
        if self.category_metadata.use_attributes:
            if self.category_metadata.attribute_label:
                predicted_labels = [attr.target.replace("\"", "") for attr in
                                    predicted_amr.attributes(role=self.category_metadata.attribute_label)]
            else:
                predicted_labels = [attr.target.replace("\"", "") for attr in predicted_amr.attributes()]
        elif use_sense:
            predicted_labels = [instance.target for instance in predicted_amr.instances()]
        else:
            predicted_labels = [strip_sense(instance.target) for instance in predicted_amr.instances()]
        return predicted_labels

    def update_error_analysis(self, graph_id, predicted,
                              target):
        predicted_labels, predicted_labels_no_sense = predicted

        # we only check senses if use_senses=True
        check_senses = predicted_labels is not None

        # we also check without senses if uses_senses=False or we're running prereqs
        if predicted_labels_no_sense is not None:
            label_found = self.find_label(predicted_labels_no_sense, target, False)
            # store the result. If we running prereqs, this sense-less version is the prereqs, otherwise it's main
            # (there's no other way of doing prereqs in NodeRecall)
            error_status = "correct" if label_found else "incorrect"
            error_version = "prereqs" if self.category_metadata.run_prerequisites else "ids"
            self.error_analysis_dict[f"{error_status}_{error_version}"].append(graph_id)
            if not label_found and check_senses:
                # if that failed no need to check with senses
                self.add_fail(graph_id)
                check_senses = False
        # if the prereqs worked and , now check for the full label if use_sense=True
        if check_senses:
            label_found = self.find_label(predicted_labels, target, True)
            error_status = "correct" if label_found else "incorrect"
            self.error_analysis_dict[f"{error_status}_ids"].append(graph_id)

    def get_predictions_for_comparison(self, predicted_amr):
        """
        Extract the relevant labels
        Args:
            predicted_amr: penman Graph
        Returns: relevant node labels, with and without senses. If either isn't needed, it's None instead.
        """
        if not self.category_metadata.use_sense or self.category_metadata.run_prerequisites:
            predicted_labels_no_sense = self._get_predicted_labels_based_on_evaluation_case(
                predicted_amr,
                use_sense=False)
        else:
            predicted_labels_no_sense = None
        # We always need them with senses if use_sense=True
        if self.category_metadata.use_sense:
            predicted_labels = self._get_predicted_labels_based_on_evaluation_case(
                predicted_amr,
                use_sense=True)
        else:
            predicted_labels = None
        return predicted_labels, predicted_labels_no_sense

    def find_label(self, predicted_labels, target_label, use_sense):
        label_found = _label_exists_in_predicted_labels(predicted_labels, target_label, use_sense)
        if not label_found and " " in target_label:
            for target_label_variant in target_label.split(" "):
                if _label_exists_in_predicted_labels(predicted_labels, target_label_variant, use_sense):
                    label_found = True
                    break
        return label_found


class PPAttachment(CategoryEvaluation):
    def __init__(self, gold_amrs, predicted_amrs, root_dir, info, predictions_directory=None):
        """
        Pragmatic attachments of ambiguous PPs
        PP Attachments come from multiple files, so if they're not already in the given graphs, we try to get them.
        """
        super().__init__(gold_amrs, predicted_amrs, root_dir, info, predictions_directory)
        # if we read in the unused PP directory instead of the whole full_cprpus, replace it with the real ones
        # These have ids pp_attachment_n
        if self.gold_amrs[0].metadata['id'].startswith(self.category_metadata.subcorpus_filename):
            print("Reading in additional files")
            self.gold_amrs, self.predicted_amrs = self.get_additional_graphs(read_in=True)
        if len(self.gold_amrs) != len(self.predicted_amrs) or len(self.gold_amrs) ==0:
            raise Exception("Different number of AMRs or 0")

    def make_results(self):
        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.gold_amrs, self.predicted_amrs)
        assert sample_size>0, "No results found!"
        rows = [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]
        self.rows.extend(rows)


class NETypeRecall(CategoryEvaluation):
    """Identifying named entity types"""

    def read_tsv(self):
        return get_2_columns_from_tsv_by_id(f"{self.corpus_path}/{self.category_metadata.tsv}")

    def update_error_analysis(self, graph_id, predictions_for_comparison, target):
        found = False
        prereq_success = False
        for edge in predictions_for_comparison.edges(role=":name"):  #.edges(role=":name"):
            entity_label = get_node_by_name(edge.source, predictions_for_comparison).target
            name_string = get_name(edge.target, predictions_for_comparison)
            if name_string == target[1]:
                self.add_prereq_success(graph_id)
                prereq_success = True
                if entity_label == target[0]:
                    self.add_success(graph_id)
                    found = True
                break
        if not found:
            self.add_fail(graph_id)
        if not prereq_success:
            self.add_prereq_fail(graph_id)


class NERecall(CategoryEvaluation):
    """Correctly creating attributes for named entities, such as the components of a name"""

    def read_tsv(self):
        return get_graphid2labels_from_tsv_file(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                         graph_id_column=self.category_metadata.graph_id_column,
                                         label_column=self.category_metadata.label_column)



    def update_error_analysis(self, graph_id, predictions_for_comparison, target):
        found = False
        entity_type = self.category_metadata.subtype
        if entity_type == "other":
            # if not name or date, try both attributes and instances
            gold_value_string = normalize_special_entity_value(target)
            for instance_or_attribute in predictions_for_comparison.instances() + predictions_for_comparison.attributes():
                # and we only need to normalise the one string
                if normalize_special_entity_value(instance_or_attribute.target) == gold_value_string:
                    self.add_success(graph_id)
                    break
        else:
            for instance in predictions_for_comparison.instances():
                if instance.target == entity_type:

                    if entity_type == "date-entity":
                        name_string = get_date_string_for_date_instance(predictions_for_comparison, instance)
                    elif entity_type == "name":
                        name_string = get_name_string_for_name_instance(predictions_for_comparison, instance)
                    # get all the relevant attributes and put them into a string of the same format as the TSV
                    if name_string == target:
                        self.add_success(graph_id)
                        found = True
                        break
        if not found:
            self.add_fail(graph_id)


class SubgraphRecall(CategoryEvaluation):
    """For multinode word meanings like "teacher" = person <- teach-01"""
    def make_results(self):
        id2subgraphs = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        recalled, sample_size = calculate_subgraph_existence_successes_and_sample_size(
            id2subgraphs, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])


class EllipsisRecall(CategoryEvaluation):
    """Find two instances of a subgraph where one is elided in the sentence"""
    def make_results(self):
        id2labels = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv)
        prereqs, recalled, sample_size = get_ellipsis_success_counts(
            id2labels, self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])


class ImperativeRecall(CategoryEvaluation):
    def make_results(self):
        id2labels = read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv, columns=[1,2,3])
        prereqs, recalled, with_correct_target, sample_size = get_imperative_success_counts(id2labels,
                                                                                            gold_amrs=self.gold_amrs,
                                                                                            predicted_amrs=self.predicted_amrs,
                                                                                            )
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [with_correct_target, sample_size])
        # self.make_results_column("Marked as imperative", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size])
        self.make_and_append_results_row("Prerequisite", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])


class WordDisambiguationRecall(CategoryEvaluation):
    def make_results(self):
        if self.category_metadata.subtype == "hand-crafted":
            fun = evaluate_word_disambiguation
        elif self.category_metadata.subtype == "bert":
            fun = evaluate_berts_mouth
        else:
            raise NotImplementedError(f"subtype {self.category_metadata.subtype} not implemented: must be bert or hand-crafted")
        self.gold_amrs, self.predicted_amrs = filter_amrs_for_name(self.category_metadata.subcorpus_filename,
                                                                   self.gold_amrs, self.predicted_amrs)
        successes, sample_size = fun(self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])


class ExactMatch(CategoryEvaluation):
    def __init__(self, gold_amrs, predicted_amrs, root_dir,
                 category_metadata, predictions_directory=None):
        super().__init__(gold_amrs, predicted_amrs, root_dir, category_metadata, predictions_directory)

        self.is_sanity_check = is_sanity_check(self.category_metadata)

        gold_amrs, predicted_amrs = filter_amrs_for_name(
            self.category_metadata.subcorpus_filename,
            self.gold_amrs,
            self.predicted_amrs)

        # Add extra graphs for deep_recursion_pronouns
        if self.extra_subcorpus_filenames is not None:
            read_in = len(self.gold_amrs) == 100  # exactly 100 graphs in deep_recursion_pronouns.txt.
            more_gold, more_pred =  self.get_additional_graphs(read_in=read_in)
            gold_amrs += more_gold
            predicted_amrs += more_pred

        self.gold_amrs = gold_amrs
        self.predicted_amrs = predicted_amrs

    def run_evaluation(self):
        if self.category_metadata.subcorpus_filename == "long_lists":
            self.long_lists()
        elif self.category_metadata.subcorpus_filename.startswith("long_lists"):
            self.long_list_sanity_check()
        else:
            self.make_results()
        return self.rows

    def make_results(self):
        successes, sample_size = compute_exact_match_successes_and_sample_size(self.gold_amrs, self.predicted_amrs,
                                                                               match_edge_labels=False,
                                                                               match_senses=False)

        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE,
                                         [successes, sample_size])
        self.make_smatch_results()

    def long_lists(self):
        conj_total_gold, conj_total_predictions, conj_true_predictions = compute_conjunct_counts(self.gold_amrs,
                                                                                                 self.predicted_amrs)

        self.make_and_append_results_row("Conjunct recall", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_gold])
        self.make_and_append_results_row("Conjunct precision", EVAL_TYPE_SUCCESS_RATE,
                                         [conj_true_predictions, conj_total_predictions])

        opi_gen_total_gold, opi_gen_total_predictions, opi_gen_true_predictions = compute_generalization_op_counts(
            self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Unseen :opi recall", EVAL_TYPE_SUCCESS_RATE,
                                         [opi_gen_true_predictions, opi_gen_total_gold])

    def long_list_sanity_check(self):
        success, sample_size = compute_exact_match_successes_and_sample_size(self.gold_amrs, self.predicted_amrs,
                                                                             match_edge_labels=False,
                                                                             match_senses=False)
        self.make_and_append_results_row("Exact match", EVAL_TYPE_SUCCESS_RATE, [success, sample_size])

