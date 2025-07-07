import argparse
import os
import pickle
import re
from typing import Tuple, Callable, List, Dict

import amconll
from penman import Graph
from vulcan.pickle_builder.pickle_builder import PickleBuilder
import penman
from amconll import parse_amconll, Entry, write_conll
from vulcan.data_handling.format_names import *

from create_vulcan_pickle import get_metadata_fieldname_and_mapper
from evaluate_single_category import get_gold_path_based_on_info
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation
from scripts.argparse_formatter import SmartFormatter
from evaluation.full_evaluation.category_evaluation.category_metadata import (category_name_to_set_class_and_metadata,
                                                                              is_testset_category,
                                                                              get_formatted_category_names_by_main_file)
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo

# Regex for SGraph sources
SOURCE_PATTERN = re.compile(r"(?P<source><[a-zA-Z0-9]+>)")


def create_pickle(gold_graphs: List[Graph], predicted_graphs: List[Graph], amconll_sentences: List[amconll.AMSentence],
                  path_to_pickle: str, extra_metadata: Dict[str, str]=None):
    """
    Read in AM Parser output files and create a vulcan-readable pickle
    Pickle contains AM dependency tree, including graph constants, and gold and predicted graphs.
    Args:
        gold_graphs: the gold graphs, read in and filtered to contain exactly the relevant ones
        predicted_graphs: the predictions, same order as gold
        amconll_sentences: the amconll sentences from the relevant files
        path_to_pickle: output path to write vulcan pickle to
    """
    # What to include in the metadata field (just ID or more)
    example_graph = gold_graphs[0]
    metadata_fieldname, metadata_mapper = get_metadata_fieldname_and_mapper(example_graph, extra_metadata=extra_metadata)

    # # read in the amconll file of parser predictions
    # with open(filtered_amconll, "r", encoding="utf-8") as f:
    #     amconll_sentences = [s for s in parse_amconll(f, False)]  # read it all in so we can close the file

    # initialise the pickle builder with the appropriate fields and their data types
    # The sentence is a table because it will stack the supertags on the words for each word
    pickle_builder = PickleBuilder({"Gold graph": FORMAT_NAME_GRAPH, "Predicted graph": FORMAT_NAME_GRAPH,
                                    "Sentence": FORMAT_NAME_OBJECT_TABLE, metadata_fieldname: FORMAT_NAME_STRING})
    added = 0

    for i, amconll_sent in enumerate(amconll_sentences):
        # just for PP attachments, there are some extra graphs in the corpus that we don't have AM conll files for
        # because they're not part of the final GrAPES dataset
        gold_amr = gold_graphs[i]
        while amconll_sent.attributes["id"] != gold_amr.metadata["id"]:
            i += 1
            gold_amr = gold_graphs[i]

        tagged_sentence = get_tagged_sentence_for_amconll_sent(amconll_sent)

        # info to put in the metadata field in the pickle (id and size if applicable)
        meta = metadata_mapper(gold_amr)
        added +=1
        # use the exact same field names as used when the pickle builder was initialised.
        pickle_builder.add_instances_by_name({"Gold graph": gold_amr,
                                              "Predicted graph": predicted_graphs[i],
                                              "Sentence": tagged_sentence,
                                              metadata_fieldname: meta
                                              })

        # add the dependency tree edges to the Sentence entry
        deptree = make_dependency_tree(amconll_sent)
        pickle_builder.add_dependency_tree_by_name("Sentence", deptree)

    # transform stored dict in pickle_builder.data into a list of dicts with original keys as value of "name"

    # write the pickle
    pickle_builder.write(path_to_pickle)
    print(f"Wrote pickle to {path_to_pickle}")
    if added != len(amconll_sentences) != len(gold_graphs) != len(predicted_graphs):
        print(f"Warning: we started with {len(amconll_sentences)} AMConll sentences and {len(gold_graphs)} graphs, but we wrote {added} to the pickle")


def get_tagged_sentence_for_amconll_sent(amconll_sent):
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
    return tagged_sentence


def create_pickle_for_error_analysis(evaluator: CategoryEvaluation, amconll_sentences: List[amconll.AMSentence],
                                     error_analysis_pickle_path: str, out_dir_path: str):

    error_eval_dict = pickle.load(open(error_analysis_pickle_path, "rb"))

    labels = evaluator.read_tsv()
    metric = evaluator.category_metadata.metric_label

    os.makedirs(out_dir_path, exist_ok=True)
    for key in error_eval_dict:
        print_key = key if not key.endswith("_id") else key[:-3]
        extra_fields = {"error status": print_key, "metric": metric}
        if len(error_eval_dict[key]) == 0:
            print("no graphs for category", key)
            continue
        golds = []
        preds = []
        amconlls = []
        for gold, pred in zip(evaluator.gold_amrs, evaluator.predicted_amrs):
            graph_id = gold.metadata["id"]
            if graph_id in error_eval_dict[key]:
                golds.append(gold)
                preds.append(pred)
                if labels:
                    extra_fields["gold label"] = labels[graph_id]
        # do these separately because for PP attachment there are extra graphs
        for amconll_sentence in amconll_sentences:
            if amconll_sentence.attributes["id"] in error_eval_dict[key]:
                amconlls.append(amconll_sentence)
        if len(golds) == 0:
            print("no matching graphs for", key)

        create_pickle(golds,preds,amconlls,
                      f"{out_dir_path}/{evaluator.category_metadata.name}_{print_key}.pickle",
                      extra_metadata=extra_fields)



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
    """AM parser makes folders called <subcorpus>_intermediary_files that contains i.a. the AM Conll file,
        mysteriously called AMR-2020_pred.amconll"""
    return f"{directory}/{subcorpus}_intermediary_files"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument("-c", "--category", help="category to evaluate Choices:" + get_formatted_category_names_by_main_file())
    parser.add_argument("-a", "--am_path", help="path to AM parser output folder: parent of all the intermediary_files folders")
    parser.add_argument("-p", "--pred_path", help="path to AM parser predicted AMR file (e.g. full_corpus.txt)")
    parser.add_argument("-g", "--gold_path", help="path to gold file (optional if GrAPES category: will use corpus/corpus.txt)", default=None)
    parser.add_argument("-o", "--output_path", help="path to write Vulcan pickle file. "
                                                    "Default error_analysis/<parser_name>/<category>_vulcan.pickle. "
                                                    "NB: If --error_analysis, we will make our own paths.", default=None)
    parser.add_argument('-n', '--parser_name', type=str,
                        help="name of parser (optional, for creating output pathname)", default="amparser")
    parser.add_argument("-e", "--error_analysis", action="store_true", help="Split up into correct and incorrect graphs according to the criteria. "
                                                                            "Only works if you have run your evaluation with the --error_analysis flag.")
    parser.add_argument("-ep", "--input_error_analysis_pickle_path",
                                     help="Path to the error analysis pickle file generated by evaluation. (Optional: if not given, will try to reconstruct from other info)",
                                     required=False, default=None)
    args = parser.parse_args()




    x: Tuple[Callable, SubcategoryMetadata] = category_name_to_set_class_and_metadata[args.category]
    eval_class, info = x
    root_dir_here = "../.."

    instance_info = EvaluationInstanceInfo(
        root_dir=root_dir_here,
        absolute_path_to_predictions_file=args.pred_path,
        absolute_path_to_gold_file=args.gold_path,
        parser_name=args.parser_name,
    )

    print(info.display_name)
    gold_path, predictions_path, use_subcorpus = get_gold_path_based_on_info(args.gold_path, info, instance_info)

    gold_amrs = penman.load(gold_path)
    predicted_amrs = penman.load(predictions_path)
    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)
    ids = dummy_evaluator.get_all_gold_ids()

    # read in the amconll file(s) of parser predictions
    amconll_sents = []
    files_to_read = ["testset" if is_testset_category(info) else info.subcorpus_filename]
    if info.extra_subcorpus_filenames is not None:
        files_to_read += info.extra_subcorpus_filenames
    for file in files_to_read:
        try:
            with open(f"{make_am_path(args.am_path, file)}/AMR-2020_pred.amconll", "r", encoding="utf-8") as f:
                amconll_sents += [s for s in parse_amconll(f, False) if s.attributes["id"] in ids]
        except FileNotFoundError as e:
            if args.category == "pp_attachment":
                pass
            else:
                print("WARNING: could not read in AM conll file for", file, e)
                raise e

    # print(len(gold_amrs), len(predicted_amrs) , len(amconll_sents))



    parent = f"{root_dir_here}/error_analysis/{instance_info.parser_name}/am_trees"
    os.makedirs(parent, exist_ok=True)

    filtered_amconll_path = f"{parent}/{info.name}.amconll"
    write_conll(filtered_amconll_path, amconll_sents)

    if args.error_analysis:
        if args.input_error_analysis_pickle_path is not None:
            input_error_analysis_pickle_path = args.input_error_analysis_pickle_path
        else:
            input_error_analysis_pickle_path = f"{root_dir_here}/error_analysis/{instance_info.parser_name}/dictionaries/{info.name}.pickle"
        out_dir = f"{parent}/vulcan_correct_and_incorrect"
        os.makedirs(out_dir, exist_ok=True)
        create_pickle_for_error_analysis(dummy_evaluator, amconll_sents, input_error_analysis_pickle_path, out_dir)
    else:
        out_dir = f"{parent}/vulcan_subcorpora"
        pickle_path = args.output_path if args.output_path is not None else f"{out_dir}/{info.name}.pickle"
        os.makedirs(out_dir, exist_ok=True)
        print(len(dummy_evaluator.gold_amrs), len(dummy_evaluator.predicted_amrs))
        create_pickle(dummy_evaluator.gold_amrs, dummy_evaluator.predicted_amrs, amconll_sents, pickle_path)
        print("wrote Vulcan pickle to", pickle_path)



