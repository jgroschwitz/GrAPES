import pickle

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
    calculate_special_entity_successes_and_sample_size
from evaluation.util import filter_amrs_for_name, get_node_name_for_gold_label, strip_sense
from evaluation.novel_corpus.word_disambiguation import evaluate_word_disambiguation


class EdgeRecall(CategoryEvaluation):
    def run_evaluation(self):
        try:
            self.make_results()
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
        return self.rows

    def make_results(self):
        prereqs, unlabeled_recalled, labeled_recalled, sample_size = calculate_edge_prereq_recall_and_sample_size_counts(
            self.category_metadata,
            gold_amrs=self.gold_amrs,
            predicted_amrs=self.predicted_amrs,
            root_dir=self.root_dir,
        )
        rows = [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [labeled_recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE,
                                      [unlabeled_recalled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]
        self.rows.extend(rows)

    def _calculate_edge_recall(self):
        id2labels = read_edge_tsv(self.root_dir, self.category_metadata)
        error_analysis = {"correct_ids": [], "incorrect_ids": [], "correct_prereqs": [], "correct_unlabelled": [],"incorrect_unlabelled": []}


        for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
            graph_id = gold_amr.metadata['id']
            if graph_id in id2labels:
                predictions_to_look_at = predicted_amr

                for target_tuple in id2labels[graph_id]:
                    self.update_error_analysis(error_analysis, graph_id, predicted_amr, target_tuple)

        if do_error_analysis:
            # write to pickle
            # TODO shuffle the error analysis lists (synchronously), so that we can get a random sample of the errors
            with open(f"{root_dir}/error_analysis/{error_analysis_output_filename}", "wb") as f:
                pickle.dump(error_analysis, f)
        assert total > 0, f"No matching graphs found! Started with {len(gold_amrs)} gold AMRs."
        return prereqs, unlabeled_recalled, recalled, total

    def update_error_analysis(self, error_analysis, graph_id, predicted, target):
        prereqs_ok = _check_prerequisites_for_edge_tuple(target, predicted)
        if prereqs_ok:
            error_analysis["correct_prereqs"].append(graph_id)
            unlabeled_edge_found = check_edge_existence(target, predicted,
                                                        match_edge_labels=False,
                                                        match_senses=self.category_metadata.use_sense)
            if unlabeled_edge_found:
                error_analysis["correct_unlabelled"].append(graph_id)

                edge_found = _check_edge_existence_with_multiple_label_options(target, predicted_amr,
                                                                               use_sense=self.category_metadata.use_sense)
                if edge_found:
                    error_analysis["correct_ids"].append(graph_id)
                else:
                    error_analysis["incorrect_ids"].append(graph_id)
            else:
                error_analysis["incorrect_unlabelled"].append(graph_id)
                error_analysis["incorrect_ids"].append(graph_id)
        else:
            error_analysis["incorrect_prereqs"].append(graph_id)
            error_analysis["incorrect_unlabelled"].append(graph_id)
            error_analysis["incorrect_ids"].append(graph_id)


class NodeRecall(CategoryEvaluation):
    def run_evaluation(self):
        self.make_results()
        # if self.category_metadata.run_prerequisites:
        #     self.make_results(prereq=True)
        return self.rows

    def make_results(self):
        success_count, prereq_success, sample_size = self.calculate_node_label_successes_and_sample_size_and_do_error_analysis()   #calculate_node_label_successes_and_sample_size(
        row = self.make_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE, [success_count, sample_size])
        self.rows.append(row)
        if self.category_metadata.run_prerequisites:
            row = self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq_success, sample_size])
            self.rows.append(row)

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

    def calculate_node_label_successes_and_sample_size_and_do_error_analysis(self):
        """
        Just a test case for generalising error analysis
        """
        print("Using new method")
        id2labels = read_label_tsv(self.root_dir, self.category_metadata.tsv)
        error_analysis = {"correct_ids": [], "incorrect_ids": [], "correct_prereqs": [], "incorrect_prereqs": []}
        for gold_amr, predicted_amr in zip(self.gold_amrs, self.predicted_amrs):
            graph_id = gold_amr.metadata['id']
            if graph_id in id2labels:
                # We want the unlabelled version for prereqs if we're doing them and
                # the main analysis if use_sense=False
                predicted_labels, predicted_labels_no_sense = self.get_predicted_labels(predicted_amr)

                # run through the labels we want to find for this graph
                for target_label in id2labels[graph_id]:
                    # start without senses if we're doing it at all, because we can skip the real analysis if it fails
                    self.update_error_analysis(error_analysis, graph_id, (predicted_labels, predicted_labels_no_sense),
                                               target_label)

        # write to pickle
        with open(f"{self.root_dir}/error_analysis/{self.category_metadata.name}.pickle", "wb") as f:
            pickle.dump(error_analysis, f)

        # get the metrics from the error analysis
        success_count = len(error_analysis["correct_ids"])
        prereq_success_count = len(error_analysis["correct_prereqs"])
        sample_size = success_count + len(error_analysis["incorrect_ids"])
        return success_count, prereq_success_count, sample_size

    def update_error_analysis(self, error_analysis, graph_id, predicted,
                              target):
        predicted_labels, predicted_labels_no_sense = predicted
        check_senses = predicted_labels is not None
        if predicted_labels_no_sense is not None:
            label_found = self.find_label(predicted_labels_no_sense, target, False)
            error_status = "correct" if label_found else "incorrect"
            error_version = "prereqs" if self.category_metadata.run_prerequisites else "ids"
            error_analysis[f"{error_status}_{error_version}"].append(graph_id)
            if not label_found and check_senses:
                # if that failed no need to check with senses
                error_analysis["incorrect_ids"].append(graph_id)
                check_senses = False
        # if the prereqs worked, now check for the full label
        if check_senses and predicted_labels is not None:
            label_found = self.find_label(predicted_labels, target, True)
            error_status = "correct" if label_found else "incorrect"
            error_analysis[f"{error_status}_ids"].append(graph_id)

    def get_predicted_labels(self, predicted_amr):
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
    def make_results(self):
        id2labels = get_2_columns_from_tsv_by_id(f"{self.corpus_path}/{self.category_metadata.tsv}")
        prereq, successes, sample_size = get_ne_type_successes_and_sample_size(
            id2labels,
            self.gold_amrs,
            self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])
        self.make_and_append_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereq, sample_size])


class NERecall(CategoryEvaluation):
    """Correctly creating attributes for named entities, such as the components of a name"""
    def make_results(self):
        id2labels_entities = get_graphid2labels_from_tsv_file(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                                              graph_id_column=self.category_metadata.graph_id_column,
                                                              label_column=self.category_metadata.label_column)
        successes, sample_size = calculate_special_entity_successes_and_sample_size(
            id2labels_entities, self.gold_amrs, self.predicted_amrs, self.category_metadata.subtype)
        self.make_and_append_results_row(self.category_metadata.metric_label, EVAL_TYPE_SUCCESS_RATE,
                                         [successes, sample_size])


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

