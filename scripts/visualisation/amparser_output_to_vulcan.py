import argparse
import pickle
import re
from typing import Tuple, Callable, List

from penman import Graph
from vulcan.pickle_builder.pickle_builder import PickleBuilder
import penman
from amconll import parse_amconll, AMSentence, Entry, write_conll
from vulcan.data_handling.format_names import *

from evaluate_single_category import SmartFormatter, get_gold_path_based_on_info
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
    get_formatted_category_names, is_testset_category
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo

# Regex for SGraph sources
SOURCE_PATTERN = re.compile(r"(?P<source><[a-zA-Z0-9]+>)")


def create_pickle(gold_graphs: List[Graph], predicted_graphs: List[Graph], filtered_amconll: str, path_to_pickle: str):
    """
    Read in AM Parser output files and create a vulcan-readable pickle
    Pickle contains AM dependency tree, including graph constants, and gold and predicted graphs.
    Args:
        gold_graphs: the gold graphs, read in and filtered to contain exactly the relevant ones
        predicted_graphs: the predictions, same order as gold
        filtered_amconll: path to the newly-written amconll file, containing exactly the same sentences in order
        path_to_pickle: output path to write vulcan pickle to
    """

    # read in the amconll file of parser predictions
    with open(filtered_amconll, "r", encoding="utf-8") as f:
        amconll_sents = [s for s in parse_amconll(f, False)]  # read it all in so we can close the file

    # initialise the pickle builder with the appropriate fields and their data types
    # The sentence is a table because it will stack the supertags on the words for each word
    pickle_builder = PickleBuilder({"Gold graph": FORMAT_NAME_GRAPH, "Predicted graph": FORMAT_NAME_GRAPH,
                                    "Sentence": FORMAT_NAME_OBJECT_TABLE, "ID": FORMAT_NAME_STRING})

    # everything is in the same order, so we can zip the lists to get all info for each corpus entry
    for gold_amr, predicted_amr, amconll_sent in zip(gold_graphs, predicted_graphs, amconll_sents):

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
                                              "Sentence": tagged_sentence,
                                              "ID": gold_amr.metadata["id"]
                                              })

        # add the dependency tree edges to the Sentence entry
        deptree = make_dependency_tree(amconll_sent)
        pickle_builder.add_dependency_tree_by_name("Sentence", deptree)

    # transform stored dict in pickle_builder.data into a list of dicts with original keys as value of "name"

    # write the pickle
    with open(path_to_pickle, "wb") as f:
        pickle_builder.write(pickle_path)


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
    parser.add_argument("-c", "--category", help="category to evaluate Choices:" + get_formatted_category_names())
    parser.add_argument("-a", "--am_path", help="path to AM parser output folder: parent of all the intermediary_files folders")
    parser.add_argument("-p", "--pred_path", help="path to AM parser predicted AMR file (e.g. full_corpus.txt)")
    parser.add_argument("-g", "--gold_path", help="path to gold file (optional if GrAPES category: will use corpus/corpus.txt)", default=None)
    parser.add_argument("-o", "--output_path", help="path to write Vulcan pickle file", default=None)
    args = parser.parse_args()


    x: Tuple[Callable, SubcategoryMetadata] = category_name_to_set_class_and_metadata[args.category]
    eval_class, info = x
    root_dir_here = "../.."

    instance_info = EvaluationInstanceInfo(
        root_dir=root_dir_here,
        absolute_path_to_predictions_file=args.pred_path,
        absolute_path_to_gold_file=args.gold_path,
    )

    print(info.display_name)
    gold_path, predictions_path, use_subcorpus = get_gold_path_based_on_info(args.gold_path, info, instance_info)

    gold_amrs = penman.load(gold_path)
    predicted_amrs = penman.load(predictions_path)

    # read in the amconll file(s) of parser predictions
    amconll_sents = []
    files_to_read = ["testset" if is_testset_category(info) else info.subcorpus_filename]
    if info.extra_subcorpus_filenames is not None:
        files_to_read += info.extra_subcorpus_filenames
    for file in files_to_read:
        try:
            with open(f"{make_am_path(args.am_path, file)}/AMR-2020_pred.amconll", "r", encoding="utf-8") as f:
                amconll_sents += [s for s in parse_amconll(f, False)]  # read it all in so we can close the file
        except FileNotFoundError as e:
            print("WARNING: could read in AM conll file for", file, e)
            raise e

    print(len(gold_amrs), len(predicted_amrs) , len(amconll_sents))

    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)

    filtered_amconll_path = f"{root_dir_here}/error_analysis/{info.name}.amconll"
    write_conll(filtered_amconll_path, amconll_sents)


    pickle_path = args.output_path if args.output_path is not None else f"{root_dir_here}/error_analysis/amparser/{info.name}_vulcan.pickle"

    create_pickle(dummy_evaluator.gold_amrs, dummy_evaluator.predicted_amrs, filtered_amconll_path, pickle_path)
    print("wrote Vulcan pickle to", pickle_path)



