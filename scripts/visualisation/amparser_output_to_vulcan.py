import argparse
import pickle
import re
from vulcan.pickle_builder.pickle_builder import PickleBuilder
import penman
from amconll import parse_amconll, AMSentence, Entry
from vulcan.data_handling.format_names import *

from evaluate_single_category import SmartFormatter
from evaluation.corpus_metrics import graph_is_in_ids
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
    get_formatted_category_names
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument("input_path", help="path to input folder")
    parser.add_argument("output_path", help="path to pickle file")
    parser.add_argument("-c", "--category", help="category to evaluate Choices:" + get_formatted_category_names())
    args = parser.parse_args()
    # create_pickle(args.input_path, args.output_path)

    eval_class, info = category_name_to_set_class_and_metadata[args.category]

    root_dir_here = "../.."

    print(info.display_name)
    category_identifier = info.subcorpus_filename if info.subcorpus_filename is not None else "testset"
    am_in_path = f"{root_dir_here}/{args.input_path}/{category_identifier}_intermediary_files"
    if category_identifier == "testset":
        gold_amrs = penman.load(f"{root_dir_here}/data/raw/gold/test.txt")
    else:
        gold_amrs = penman.load(f"{root_dir_here}/corpus/subcorpora/{category_identifier}.txt")
    predictions_directory = f"{args.input_path}"
    predicted_amrs = penman.load(f"{predictions_directory}/{category_identifier}.txt")
    # read in the amconll file of parser predictions
    # with open(f"{in_path}/AMR-2020_pred.amconll", "r", encoding="utf-8") as f:
    #     amconll_sents = [s for s in parse_amconll(f, False)]  # read it all in so we can close the file
    # print(len(gold_amrs), len(predicted_amrs) , len(amconll_sents))

    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, root_dir_here, info, predictions_directory)
    print(len(dummy_evaluator.gold_amrs), len(dummy_evaluator.predicted_amrs))
    filtered_golds, filtered_preds = dummy_evaluator.filter_graphs()
    print(len(filtered_golds), len(filtered_preds))

    # if info.tsv is not None:
    #     filtered = []
    #     # The relevant graph IDS
    #     ids = read_label_tsv("../..", info.tsv, graph_id_column=info.graph_id_column).keys()
    #     for gold_amr, predicted_amr, amconll_sent in zip(gold_amrs, predicted_amrs, amconll_sents):
    #         if graph_is_in_ids(gold_amr, ids):
    #             if amconll_sent.attributes["id"] in ids:
    #                 filtered.append((gold_amr, predicted_amr,amconll_sent))
    #             else:
    #                 raise ValueError(f"amconll_sent.attributes['id'] not in ids: {amconll_sent.attributes['id']}")
    #     print(len(filtered))
    #
    #
    # else:
    #     filtered_predicted_amrs = filter_amrs_for_name(category_identifier, gold_amrs, predicted_amrs)
    #     print(len(filtered_predicted_amrs[0]), len(amconll_sents))
