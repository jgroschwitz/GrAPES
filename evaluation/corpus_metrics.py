import pickle
from typing import Counter, List, Union, Tuple
import tempfile

import penman
import smatch
from penman import Graph
from penman import load

from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.util import get_node_name_for_gold_label, strip_sense
from evaluation.graph_matcher import equals_modulo_isomorphy, check_fragment_existence
from evaluation.file_utils import read_node_label_tsv, load_corpus_from_folder, read_edge_tsv, get_graph_for_node_string


def compute_precision_recall_f1_from_counters(pred_counter: Counter, gold_counter: Counter):
    total_predictions = sum(pred_counter.values())
    total_gold = sum(gold_counter.values())
    true_predictions = 0
    for key in pred_counter.keys():
        true_predictions += min(pred_counter[key], gold_counter[key])
    return compute_precision_recall_f1(true_predictions, total_predictions, total_gold)


def compute_precision_recall_f1_from_counter_lists(pred_counters: List[Counter], gold_counters: List[Counter]):
    total_gold, total_predictions, true_predictions = compute_correctness_counts_from_counter_lists(gold_counters,
                                                                                                    pred_counters)
    return compute_precision_recall_f1(true_predictions, total_predictions, total_gold)


def compute_correctness_counts_from_counter_lists(gold_counters, pred_counters):
    total_predictions = sum([sum(c.values()) for c in pred_counters])
    total_gold = sum([sum(c.values()) for c in gold_counters])
    true_predictions = 0
    for pred_counter, gold_counter in zip(pred_counters, gold_counters):
        for key in pred_counter.keys():
            true_predictions += min(pred_counter[key], gold_counter[key])
    return total_gold, total_predictions, true_predictions


def compute_precision_recall_f1(true_predictions, total_predictions, total_gold):
    if total_predictions > 0:
        precision = true_predictions / total_predictions
    else:
        precision = 0
    recall = true_predictions / total_gold
    if true_predictions == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_exact_match_fraction(golds: List[Graph], predictions: List[Graph],
                                 match_edge_labels: bool = True, match_senses: bool = True):
    correct, sample_size = compute_exact_match_successes_and_sample_size(golds, predictions,
                                                                         match_edge_labels=match_edge_labels,
                                                                         match_senses=match_senses)
    return correct / sample_size


def compute_exact_match_successes_and_sample_size(golds: List[Graph], predictions: List[Graph],
                                                  match_edge_labels: bool = True, match_senses: bool = True):
    assert len(golds) == len(predictions)
    correct = 0
    for gold, prediction in zip(golds, predictions):
        if equals_modulo_isomorphy(gold, prediction, match_edge_labels=match_edge_labels, match_senses=match_senses):
            correct += 1
        # else:
        #     print("Gold:")
        #     print(penman.encode(gold))
        #     print("Prediction:")
        #     print(penman.encode(prediction))
        #     print()
    return correct, len(golds)


def compute_smatch_f(gold_file_path: str, prediction_file_path: str):
    with open(gold_file_path) as gold_f:
        with open(prediction_file_path) as prediction_f:
            # the function returns a generator that generates just one item (a triple p, r, f), so we use next to get it
            p, r, f = next(smatch.score_amr_pairs(gold_f, prediction_f))

    return p, r, f


def compute_smatch_f_from_graph_lists(gold_graphs: List[Graph], predicted_graphs: List[Graph]):
    # write graphs into temporary files, using pythons inbuilt function for temp files
    with tempfile.NamedTemporaryFile(mode="w") as gold_f:
        with tempfile.NamedTemporaryFile(mode="w") as prediction_f:
            penman.dump(gold_graphs, gold_f)
            penman.dump(predicted_graphs, prediction_f)
            return compute_smatch_f(gold_f.name, prediction_f.name)[2]


def calculate_node_label_recall(category_metadata: SubcategoryMetadata, gold_amrs=None, predicted_amrs=None,
                                parser_name=None,
                                root_dir="../../",
                                prereq=False,
                                error_analysis_output_filename=None, error_analysis_message=None):
    success_count, sample_size = calculate_node_label_successes_and_sample_size(category_metadata, gold_amrs,
                                                                                predicted_amrs, root_dir,
                                                                                prereq=prereq,
                                                                                error_analysis_output_filename=error_analysis_output_filename,
                                                                                error_analysis_message=error_analysis_message)

    recall = success_count / sample_size if sample_size > 0 else 1.0
    return recall


def calculate_node_label_successes_and_sample_size(category_metadata: SubcategoryMetadata,
                                                   gold_amrs=None,
                                                   predicted_amrs=None,
                                                   root_dir="../../",
                                                   prereq=False,
                                                   error_analysis_output_filename=None, error_analysis_message=None):
    if prereq:
        use_sense = category_metadata.use_sense_prereq
    else:
        use_sense = category_metadata.use_sense
    id2labels = read_node_label_tsv(root_dir, category_metadata.tsv)
    success_count = 0
    sample_size = 0
    do_error_analysis = error_analysis_output_filename is not None
    error_analysis = {"gold_graphs": [], "predicted_graphs": [], "sentences": [], "highlights": [],
                      "message": error_analysis_message}
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels:
            sample_size += len(id2labels[gold_amr.metadata['id']])
            predicted_labels = _get_predicted_labels_based_on_evaluation_case(category_metadata.attribute_label, predicted_amr,
                                                                              category_metadata.use_attributes, use_sense=use_sense)
            for target_label in id2labels[gold_amr.metadata['id']]:
                # print(label)
                # print(gold_amr.instances())
                label_found = _label_exists_in_predicted_labels(predicted_labels, target_label, use_sense)
                if not label_found and " " in target_label:
                    for target_label_variant in target_label.split(" "):
                        if _label_exists_in_predicted_labels(predicted_labels, target_label_variant, use_sense):
                            label_found = True
                            break
                if label_found:
                    success_count += 1
                elif do_error_analysis:
                    missing_label = get_node_name_for_gold_label(target_label, gold_amr, category_metadata.attribute_label)
                    error_analysis["gold_graphs"].append(gold_amr)
                    error_analysis["predicted_graphs"].append(predicted_amr)
                    error_analysis["sentences"].append(gold_amr.metadata['snt'])
                    error_analysis["highlights"].append([missing_label])
    if do_error_analysis:
        # write to pickle
        # TODO shuffle the error analysis lists (synchronously), so that we can get a random sample of the errors
        with open(f"{root_dir}/error_analysis/{error_analysis_output_filename}", "wb") as f:
            pickle.dump(error_analysis, f)
    return success_count, sample_size


def calculate_subgraph_existence_successes_and_sample_size(id2subgraphs, gold_amrs: List[Graph],
                                                           predicted_amrs: List[Graph]):
    success_count = 0
    sample_size = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2subgraphs:
            sample_size += len(id2subgraphs[gold_amr.metadata['id']])
            for subgraph_string in id2subgraphs[gold_amr.metadata['id']]:
                if check_fragment_existence(subgraph_string, predicted_amr):
                    success_count += 1
    return success_count, sample_size


def _get_predicted_labels_based_on_evaluation_case(attribute_label, predicted_amr, use_attributes, use_sense):
    """
    Get the instances or attributes in the given predicted AMR
    Note that if use_attributes and use_sense are both true, we get the attributes, not the senses.
        If they are both false, we get the instances without their senses.
    :param attribute_label: if not None, get all attributes in predicted_amr with this label
    :param predicted_amr: AMR to search through
    :param use_attributes: if True, get all attributes (restricted to attribute_label if given)
    :param use_sense: if True, get all instances with their senses; otherwise all instances without their senses
    :return: list of either attributes or senses (not both)
    """
    if use_attributes:
        if attribute_label:
            predicted_labels = [attr.target.replace("\"", "") for attr in
                                predicted_amr.attributes(role=attribute_label)]
        else:
            predicted_labels = [attr.target.replace("\"", "") for attr in predicted_amr.attributes()]
    elif use_sense:
        predicted_labels = [instance.target for instance in predicted_amr.instances()]
    else:
        predicted_labels = [strip_sense(instance.target) for instance in predicted_amr.instances()]
    return predicted_labels


def run_checks_and_get_backup_data_if_applicable(attribute_label, gold_amrs, parser_name, predicted_amrs, root_dir,
                                                 use_attributes, use_sense):
    assert predicted_amrs is not None or parser_name is not None
    if not use_sense:
        assert not use_attributes  # removing senses for attributes does not make sense
    if attribute_label:
        assert use_attributes
    if gold_amrs is None:
        gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")
    if predicted_amrs is None:
        predicted_amrs = load(f"{root_dir}/{parser_name}-output/testset.txt")
    return gold_amrs, predicted_amrs


def calculate_edge_recall_for_tsv_file(subcategory_info: SubcategoryMetadata, gold_amrs=None, predicted_amrs=None, parser_name=None,
                                       root_dir="../../",
                                       error_analysis_output_filename=None, error_analysis_message=None,
                                       ):
    prereq_successes, unlabeled_successes, recall_successes, sample_size = calculate_edge_prereq_recall_and_sample_size_counts(
        subcategory_info,
        gold_amrs, predicted_amrs,
        root_dir,
        error_analysis_output_filename, error_analysis_message
        )
    prereq = prereq_successes / sample_size if sample_size > 0 else 1.0
    recall = recall_successes / sample_size if sample_size > 0 else 1.0
    return prereq, recall


def calculate_edge_prereq_recall_and_sample_size_counts(subcategory_info,
                                                        gold_amrs=None,
                                                        predicted_amrs=None,
                                                        root_dir="../../",
                                                        error_analysis_output_filename=None,
                                                        error_analysis_message=None,
                                                        ):
    """
    Returns prereq_successes, unlabeled_successes, recall_successes, sample_size
    """
    # predicted_amrs = gold_amrs  # this is for debugging: check if the gold matches what is written in the tsv file
    #  (both recall and prerequisites should be 1.0, except if the gold graph has an error and the tsv is correct)
    id2labels = read_edge_tsv(root_dir, subcategory_info=subcategory_info)
    print(f"Found {len(id2labels)} items")
    prereq_successes, unlabeled_successes, recall_successes, sample_size = _calculate_edge_recall(
        error_analysis_message,
        error_analysis_output_filename,
        gold_amrs,
        id2labels,
        predicted_amrs,
        root_dir,
        subcategory_info.use_sense)
    print(prereq_successes, unlabeled_successes, recall_successes, sample_size)
    return prereq_successes, unlabeled_successes, recall_successes, sample_size


def _calculate_edge_recall(error_analysis_message, error_analysis_output_filename, gold_amrs, id2labels, predicted_amrs,
                           root_dir, use_sense):
    assert len(gold_amrs) > 0 and len(gold_amrs) == len(predicted_amrs), f"We have {len(gold_amrs)} gold AMRs and {len(predicted_amrs)} predicted AMRs"
    recalled = 0
    unlabeled_recalled = 0
    prereqs = 0
    total = 0
    do_error_analysis = error_analysis_output_filename is not None
    error_analysis = {"gold_graphs": [], "predicted_graphs": [], "sentences": [], "highlights": [],
                      "message": error_analysis_message}
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels:
            total += len(id2labels[gold_amr.metadata['id']])

            for target_tuple in id2labels[gold_amr.metadata['id']]:
                if _check_prerequisites_for_edge_tuple(target_tuple, predicted_amr):
                    prereqs += 1
                    unlabeled_edge_found = check_edge_existence(target_tuple, predicted_amr, match_edge_labels=False,
                                                                match_senses=use_sense)
                    if unlabeled_edge_found:
                        unlabeled_recalled += 1
                    edge_found = _check_edge_existence_with_multiple_label_options(target_tuple, predicted_amr,
                                                                                   use_sense=use_sense)
                    if edge_found:
                        recalled += 1
                # else:
                #     print("Prerequisites not met for edge tuple: " + str(target_tuple))
                # elif do_error_analysis:
                # TODO error analysis for edges
                #     missing_labels = get_node_names_for_gold_edge_specification(target_tuple, gold_amr)
                #     error_analysis["gold_graphs"].append(gold_amr)
                #     error_analysis["predicted_graphs"].append(predicted_amr)
                #     error_analysis["sentences"].append(gold_amr.metadata['snt'])
                #     error_analysis["highlights"].append(missing_labels)
    if do_error_analysis:
        # write to pickle
        # TODO shuffle the error analysis lists (synchronously), so that we can get a random sample of the errors
        with open(f"{root_dir}/error_analysis/{error_analysis_output_filename}", "wb") as f:
            pickle.dump(error_analysis, f)
    assert total > 0, f"No matching graphs found! Started with {len(gold_amrs)} gold AMRs."
    return prereqs, unlabeled_recalled, recalled, total


def _check_edge_existence_with_multiple_label_options(target_triple, predicted_amr, use_sense):
    edge_found = check_edge_existence(target_triple, predicted_amr, match_senses=use_sense)
    if not edge_found and " " in target_triple[1]:
        for edge_label_variant in target_triple[1].split(" "):
            new_tuple_as_list = list(target_triple)
            new_tuple_as_list[1] = edge_label_variant
            # noinspection PyTypeChecker
            if check_edge_existence(tuple(new_tuple_as_list), predicted_amr, match_senses=use_sense):
                edge_found = True
                break
    return edge_found


def _label_exists_in_predicted_labels(predicted_labels, target_label, use_sense):
    target_label = target_label.replace("\"", "")
    if not use_sense:
        target_label = strip_sense(target_label)
    return target_label in predicted_labels


def _check_prerequisites_for_edge_tuple(edge_tuple, predicted_amr):
    if edge_tuple_contains_secondary_parent_information(edge_tuple):
        return check_fragment_existence(edge_tuple[0], predicted_amr) and \
            check_fragment_existence(edge_tuple[2], predicted_amr) and \
            check_fragment_existence(edge_tuple[3], predicted_amr)
    else:
        return check_fragment_existence(edge_tuple[0], predicted_amr) and \
            check_fragment_existence(edge_tuple[2], predicted_amr)


def edge_tuple_contains_secondary_parent_information(edge_tuple):
    return len(edge_tuple) == 5


def check_edge_existence(edge_tuple_from_tsv: Union[Tuple[str, str, str], Tuple[str, str, str, str, str]],
                         predicted_amr, match_edge_labels: bool = True, match_senses: bool = True) -> bool:
    """
    Checks if the given edge (specified by a triple in the format we use in tsv files) exists in the predicted AMR.

    :param match_senses:
    :param match_edge_labels: If false, all edge labels will be ignored (i.e. also the mapping of the source/target
        graphs of the edge will be less precise).
    :param edge_tuple_from_tsv: A triple where the first and third entries are "node_strings", from the tsv file,
    i.e. either node labels or penman graph linearizations. They are the source and target of the edge, respectively.
    The second entry in the triple is the edge label. Can also be a quintuple, with additional entries for a secondary
    parent and its edge label.
    :param predicted_amr:
    :return:
    """
    source_graph = get_graph_for_node_string(edge_tuple_from_tsv[0])
    target_graph = get_graph_for_node_string(edge_tuple_from_tsv[2])
    connected_graph_string = penman.encode(source_graph)[:-1]  # remove closing bracket
    # add edge, target graph
    connected_graph_string += f" {edge_tuple_from_tsv[1]} {penman.encode(target_graph)}"
    # add secondary parent, if present
    if edge_tuple_contains_secondary_parent_information(edge_tuple_from_tsv):
        # attach secondary parent to the target graph, so we need to remove the closing bracket of the target graph
        # first.
        connected_graph_string = connected_graph_string[
                                 :-1] + f" {invert_edge_label(edge_tuple_from_tsv[4])} {penman.encode(get_graph_for_node_string(edge_tuple_from_tsv[3]))})"
    connected_graph_string += ")"
    # TODO would like to have unlabeled version as well, but that won't work with this trick; needs rework
    #  -- EDIT may work now? Ignoring all edge labels in the test may be imprecise, but good enough.
    # print(connected_graph_string)
    ret = check_fragment_existence(connected_graph_string, predicted_amr, match_edge_labels, match_senses)
    # print(ret)
    # if not ret:
    #     print(connected_graph_string)
    return ret


def invert_edge_label(edge_label: str) -> str:
    if edge_label.endswith("-of") and not edge_label == "consist-of":
        return edge_label[:-3]
    else:
        return edge_label + "-of"
