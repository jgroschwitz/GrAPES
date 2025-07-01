import argparse
import pickle
import re
from typing import Tuple, Callable

from nltk.sem.logic import AnyType
from vulcan.pickle_builder.pickle_builder import PickleBuilder
import penman
from amconll import parse_amconll, AMSentence, Entry, write_conll
from vulcan.data_handling.format_names import *

from create_vulcan_pickle import get_am_dependency_trees
from evaluate_single_category import SmartFormatter
from evaluation.corpus_metrics import graph_is_in_ids
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
    get_formatted_category_names
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.full_evaluation.run_full_evaluation import root_dir_here
from evaluation.util import filter_amrs_for_name

# Regex for SGraph sources
SOURCE_PATTERN = re.compile(r"(?P<source><[a-zA-Z0-9]+>)")

def create_pickle(path_to_parser_output_folder: str, path_to_pickle: str):
    """
    Read in AM Parser output files and create a vulcan-readable pickle
    Pickle contains AM dependency tree, including graph constants, and gold and predicted graphs.
    Args:
        path_to_parser_output_folder: Path to parent folder containing edge_label_scores.txt etc
        path_to_pickle: Where you want to write the pickle, including the pickle file name
    """
    # read in AMRs
    gold_amrs = penman.load(f"{path_to_parser_output_folder}/goldAMR.txt")
    predicted_amrs = penman.load(f"{path_to_parser_output_folder}/parserOut.txt")

    # read in the amconll file of parser predictions
    with open(f"{path_to_parser_output_folder}/AMR-2020_pred.amconll", "r", encoding="utf-8") as f:
        amconll_sents = [s for s in parse_amconll(f)]  # read it all in so we can close the file

    # initialise the pickle builder with the appropriate fields and their data types
    # The sentence is a table because it will stack the supertags on the words for each word
    pickle_builder = PickleBuilder({"Gold graph": FORMAT_NAME_GRAPH, "Predicted graph": FORMAT_NAME_GRAPH,
                                    "Sentence": FORMAT_NAME_OBJECT_TABLE})

    # everything is in the same order, so we can zip the lists to get all info for each corpus entry
    for gold_amr, predicted_amr, amconll_sent in zip(gold_amrs, predicted_amrs, amconll_sents):

        # list of lists of pairs (datatype, content)
        # each word is a list of the entries for the table, paired with their data type:
        # e.g. [("token", "dog"),("graph", <graph for dog>)]
        tagged_sentence = []

        for entry in amconll_sent.words:
            tagged_token = []
            tagged_sentence.append(tagged_token)

            tagged_token.append((FORMAT_NAME_TOKEN, entry.token))
            if entry.fragment == "_":
                # empty graph constants are treated by Vulcan as tokens, not graphs
                tagged_token.append((FORMAT_NAME_TOKEN, entry.fragment))
            else:
                # relexicalise the delexicalised graph constant
                tagged_token.append((FORMAT_NAME_GRAPH_STRING, relabel_supertag(entry.fragment, entry)))

        # use the exact same field names as used when the pickle builder was initialised.
        pickle_builder.add_instances_by_name({"Gold graph": gold_amr,
                                              "Predicted graph": predicted_amr,
                                              "Sentence": tagged_sentence})

        # add the dependency tree edges to the Sentence entry
        deptree = make_dependency_tree(amconll_sent)
        pickle_builder.add_dependency_tree_by_name("Sentence", deptree)

    # transform stored dict in pickle_builder.data into a list of dicts with original keys as value of "name"
    final_data = pickle_builder.make_data_for_pickle()

    # write the pickle
    with open(path_to_pickle, "wb") as f:
        pickle.dump(final_data, f)


def make_dependency_tree(amconll_sent):
    """
    Given an amconll sentence, return a list of edges (source index, target index, label)
    :param amconll_sent: amconll.AMSentence: parsed sentence.
    :return: list of triples (int, int, str)
    """
    ret = []
    for i, entry in enumerate(amconll_sent.words):
        if entry.label not in ["IGNORE", "ROOT"]:
            # TODO why are you subtracting 1?
            ret.append((entry.head - 1, i, entry.label))
    ret += [(-1, i, "ROOT") for i, entry in enumerate(amconll_sent.words) if entry.label == "ROOT"]
    return ret


def relabel_supertag(supertag, amconll_entry: Entry):
    """
    Relexicalise a graph fragment, replacing --LEX-- with the correct label from the amconll entry.
    Args:
        supertag: str: the graph in string form.
        amconll_entry: amconll.Entry from which the new label is taken

    Returns:
        The updated graph in string form, also without its <root> source label.
    """
    if supertag == "_" or supertag == "NONE":
        return supertag
    else:
        supertag = supertag.replace("--LEX--", amconll_entry.lexlabel).replace("$LEMMA$", amconll_entry.lemma) \
            .replace("$FORM$", amconll_entry.token)
        supertag = supertag.replace("<root>", "")
        return SOURCE_PATTERN.sub(r" / \g<source>", supertag)

def make_am_path(directory, subcorpus):
    return f"{directory}/{subcorpus}_intermediary_files"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument("input_path", help="path to input folder")
    parser.add_argument("output_path", help="path to pickle file")
    parser.add_argument("-c", "--category", help="category to evaluate Choices:" + get_formatted_category_names())
    args = parser.parse_args()
    # create_pickle(args.input_path, args.output_path)

    x: Tuple[Callable, SubcategoryMetadata] = category_name_to_set_class_and_metadata[args.category]
    eval_class, info = x
    root_dir_here = "../.."

    print(info.display_name)
    category_identifier = info.subcorpus_filename if info.subcorpus_filename is not None else "testset"
    if category_identifier == "testset":
        gold_amrs = penman.load(f"{root_dir_here}/data/raw/gold/test.txt")
    else:
        gold_amrs = penman.load(f"{root_dir_here}/corpus/subcorpora/{category_identifier}.txt")
    predictions_directory = f"{args.input_path}"
    predicted_amrs = penman.load(f"{predictions_directory}/{category_identifier}.txt")

    # read in the amconll file(s) of parser predictions
    amconll_sents = []
    files_to_read = [category_identifier]
    if info.extra_subcorpus_filenames is not None:
        files_to_read += info.extra_subcorpus_filenames
    for file in files_to_read:
        try:
            with open(f"{make_am_path(args.input_path, file)}/AMR-2020_pred.amconll", "r", encoding="utf-8") as f:
                amconll_sents += [s for s in parse_amconll(f, False)]  # read it all in so we can close the file
        except FileNotFoundError as e:
            print("WARNING: could read in AM conll file for", file, e)

    print(len(gold_amrs), len(predicted_amrs) , len(amconll_sents))

    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, root_dir_here, info, predictions_directory)
    print(len(dummy_evaluator.gold_amrs), len(dummy_evaluator.predicted_amrs))
    filtered_golds, filtered_preds = dummy_evaluator.filter_graphs()
    print(len(filtered_golds), len(filtered_preds), len(amconll_sents))

    ids = dummy_evaluator.get_all_gold_ids()
    print(len(ids), len(filtered_golds), len(filtered_preds), len(amconll_sents))

    write_conll(f"{root_dir_here}/error_analysis/{category_identifier}.amconll", amconll_sents)
    trees, am_sents = get_am_dependency_trees(f"{root_dir_here}/error_analysis/{category_identifier}.amconll", ids)

    if not (len(filtered_golds) == len(amconll_sents)):
        print("WARNING: different lengths of trees and graphs.")
        gold_pred_am_id = []
        for i in range(len(ids)):
            graph_id = ids[i]
            gold = filtered_golds[i]
            pred = filtered_preds[i]
            found = False
            for t, s in zip(trees, am_sents):
                if s.attributes["id"] == graph_id:
                    found = True
                    break
            if not found:
                t = []
            gold_pred_am_id.append((gold, pred, t, graph_id))
    else:
        gold_pred_am_id = zip(filtered_golds, filtered_preds, am_sents, ids)

    print(len(gold_pred_am_id))

    pickle_builder = PickleBuilder({"Gold graph": FORMAT_NAME_GRAPH,
                                    "Predicted graph": FORMAT_NAME_GRAPH,
                                    "Sentence": FORMAT_NAME_OBJECT_TABLE,
                                    "ID": FORMAT_NAME_STRING})

    # everything is in the same order, so we can zip the lists to get all info for each corpus entry
    for gold_amr, predicted_amr, t, graph_id in gold_pred_am_id:
        # use the exact same field names as used when the pickle builder was initialised.
        pickle_builder.add_instances_by_name({"Gold graph": gold_amr,
                                              "Predicted graph": predicted_amr,
                                              "Sentence": t,
                                             "ID": graph_id
                                              })

        # if amconll_sent is not None:
        #     # add the dependency tree edges to the Sentence entry
        #     deptree = make_dependency_tree(amconll_sent)
        #     pickle_builder.add_dependency_tree_by_name("Sentence", deptree)

    pickle_path = f"{root_dir_here}/error_analysis/{args.category}.pickle"
    pickle_builder.write(pickle_path)


