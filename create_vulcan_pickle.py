import argparse
import os
import pickle
import re
from typing import Tuple, Callable, List

import penman
from penman import load, Graph
from vulcan.data_handling.format_names import FORMAT_NAME_TOKENIZED_STRING, FORMAT_NAME_GRAPH, FORMAT_NAME_STRING
from vulcan.pickle_builder.pickle_builder import PickleBuilder

from evaluate_single_category import get_gold_path_based_on_info
from evaluation.file_utils import read_edge_tsv, read_tsv_with_comments
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, size_mappers
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
    get_categories_by_main_file, is_testset_category
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo


def make_dummy_evaluator_given_graphs(gold_amrs: List[Graph], predicted_amrs: List[Graph],
                                      subcategory_name: str, parser_name: str=None):
    """
    Used when we know we don't need to read in any additional files
    Args:
        gold_amrs:
        predicted_amrs:
        subcategory_name:
        parser_name:

    Returns:

    """
    eval_class, info = category_name_to_set_class_and_metadata[subcategory_name]
    instance_info = EvaluationInstanceInfo(
        do_error_analysis=True,
    )
    if parser_name is not None:
        instance_info.parser_name = parser_name
    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)
    return dummy_evaluator


def make_dummy_evaluator(prediction_path, gold_path, subcategory_name, parser_name=None):
    eval_class, info = category_name_to_set_class_and_metadata[subcategory_name]

    instance_info = EvaluationInstanceInfo(
        absolute_path_to_predictions_file=prediction_path,
        absolute_path_to_gold_file=gold_path,
        do_error_analysis=True,
    )
    if parser_name is not None:
        instance_info.parser_name = parser_name

    gold_path, predictions_path, use_subcorpus = get_gold_path_based_on_info(gold_path, info, instance_info)

    gold_amrs = penman.load(gold_path)
    predicted_amrs = penman.load(predictions_path)

    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)
    return dummy_evaluator


def get_size(gold_graph, mapper):
    size0 = int(gold_graph.metadata["size0"])
    return mapper(size0)


def create_pickle(gold_graphs: List[Graph], predicted_graphs: List[Graph], path_to_pickle: str, extra_metadata=None):
    """
    Create a vulcan-readable pickle
    Pickle contains gold and predicted graphs, sentence, graph ID, and, if size is defined, the size
    Args:
        gold_graphs: the gold graphs, read in and filtered to contain exactly the relevant ones
        predicted_graphs: the predictions, same order as gold
        path_to_pickle: output path to write vulcan pickle to
        extra_metadata: Dict str: str of anything else you want printed along with the sentence ID (and size for Structural Generalisation)
    """

    example_graph = gold_graphs[0]
    metadata_fieldname, metadata_mapper = get_metadata_fieldname_and_mapper(example_graph, extra_metadata=extra_metadata)


    # initialise the pickle builder with the appropriate fields and their data types
    # The sentence is a table because it will stack the supertags on the words for each word
    pickle_builder = PickleBuilder({"Gold graph": FORMAT_NAME_GRAPH, "Predicted graph": FORMAT_NAME_GRAPH,
                                    "Sentence": FORMAT_NAME_STRING, metadata_fieldname: FORMAT_NAME_STRING})


    # everything is in the same order, so we can zip the lists to get all info for each corpus entry
    for gold_amr, predicted_amr in zip(gold_graphs, predicted_graphs):
        meta = metadata_mapper(gold_amr)

        # use the exact same field names as used when the pickle builder was initialised.
        pickle_builder.add_instances_by_name({"Gold graph": gold_amr,
                                              "Predicted graph": predicted_amr,
                                              "Sentence": gold_amr.metadata["snt"],
                                              metadata_fieldname: meta})

    # write the pickle
    pickle_builder.write(path_to_pickle)
    print("Wrote pickle to", path_to_pickle)


def create_pickle_for_error_analysis(evaluator: CategoryEvaluation, error_analysis_pickle_path: str, out_path: str):
    error_eval_dict = pickle.load(open(error_analysis_pickle_path, "rb"))
    out_dir = f"{out_path}/{evaluator.instance_info.parser_name}/vulcan_correct_and_incorrect"
    os.makedirs(out_dir, exist_ok=True)
    for key in error_eval_dict:
        if len(error_eval_dict[key]) == 0:
            print("no graphs for category", key)
            continue
        golds = []
        preds = []
        for gold, pred in zip(evaluator.gold_amrs, evaluator.predicted_amrs):
            graph_id = gold.metadata["id"]
            if graph_id in error_eval_dict[key]:
                golds.append(gold)
                preds.append(pred)
        if len(golds) == 0:
            print("no matching graphs for", key)
        create_pickle(golds, preds, f"{out_dir}/{evaluator.category_metadata.name}_{key}.pickle", extra_metadata={"": key})


if __name__ == "__main__":
    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("-p", "--predictions_path", help="Path to the predictions file", required=True)
    command_line_parser.add_argument("-g", "--gold_path", default=None, help="Path to the gold file. Optional for GrAPES categories (will use corpus/corpus.txt)", required=False)
    command_line_parser.add_argument("-o", "--output_path", help="Path to the output folder", default="error_analysis")
    command_line_parser.add_argument("-c", "--category", help="Name of the category to make a pickl for",
                                     required=False, default=None)
    command_line_parser.add_argument("-ep", "--error_analysis_pickle_path",
                                     help="Path to the error analysis pickle file. (Optional: if not given, will try to reconstruct from other info)",
                                     required=False, default=None)
    command_line_parser.add_argument("-e", "--error_analysis",
                                     action="store_true",
                                     help="Flag: if used, graphs will be split according to correctness using pickled "
                                          "results dictionaries. If -ep is not also used, will look in a default"
                                          " location using parser name and category.")
    command_line_parser.add_argument('-n', '--parser_name', type=str,
                        help="name of parser (optional)", default="parser", required=False)

    args = command_line_parser.parse_args()
    pickle_dir = args.output_path
    os.makedirs(pickle_dir, exist_ok=True)

    if args.gold_path is None:
        gold_path = "corpus/corpus.txt"
    else:
        gold_path = args.gold_path

    if args.category is None and not args.error_analysis:
        print("Doing all")
        pickle_path = f"{pickle_dir}/{args.parser_name}/{os.path.basename(args.predictions_path)[:-4]}_vulcan.pickle"
        golds = penman.load(gold_path)
        preds = penman.load(args.predictions_path)
        create_pickle(golds, preds, pickle_path)
    else:

        if args.error_analysis:
            print("Error analysis")
            if args.error_analysis_pickle_path is None:
                if args.parser_name is None:
                    print("parser_name must be provided if no error analysis pickle path is given")
                    exit(1)
                else:
                    error_analysis_pickle_dir = f"error_analysis/{args.parser_name}/dictionaries"
                    error_analysis_pickle_path = None

            else:
                error_analysis_pickle_path = args.error_analysis_pickle_path
                error_analysis_pickle_dir = None

            if args.category is None:
                if error_analysis_pickle_dir is None:
                    print("category must be provided if no error analysis pickle path is given")
                    exit(1)
                print("Making pickles for all categories in given file")
                gold_amrs = load(gold_path)
                predicted_amrs = load(args.predictions_path)

                for category in category_name_to_set_class_and_metadata:
                    try:
                        evaluator = make_dummy_evaluator_given_graphs(gold_amrs, predicted_amrs,
                                                         category, args.parser_name)
                        create_pickle_for_error_analysis(
                            evaluator,
                            f"{error_analysis_pickle_dir}/{category}.pickle",
                            args.output_path)
                    except:
                        pass
            else:
                evaluator = make_dummy_evaluator(args.predictions_path, gold_path,
                                                 args.category, args.parser_name)
                if error_analysis_pickle_path is None:
                    error_analysis_pickle_path = f"{error_analysis_pickle_dir}/{args.category}.pickle"
                create_pickle_for_error_analysis(
                    evaluator,
                    error_analysis_pickle_path,
                    args.output_path)

        else:
            pickle_path = f"{pickle_dir}/{args.category_name}_vulcan.pickle"
            evaluator = make_dummy_evaluator(args.predictions_path, gold_path,
                                             args.category, args.parser_name)
            create_pickle(evaluator.gold_amrs, evaluator.predicted_amrs, pickle_path)


def get_metadata_fieldname_and_mapper(graph, extra_metadata=None):
    if extra_metadata is None:
        ending = ""
        metadata_fieldname = "ID"
    else:
        ending = " ".join([f"{key}: {val}" for key, val in extra_metadata.items()])
        metadata_fieldname = "MetaData"
    graph_id = graph.metadata["id"]
    parts = graph_id.split("_")
    prefix = "_".join(parts[:-1])
    if prefix in size_mappers:
        mapper = size_mappers[prefix]
        metadata_fieldname = "MetaData"
        metadata_maker = lambda gold_amr: f'ID: {gold_amr.metadata["id"]}  Size: {get_size(gold_amr, mapper)}  {ending}'
    else:
        metadata_maker = lambda gold_amr: f'{gold_amr.metadata["id"]}  {ending}'

    return metadata_fieldname, metadata_maker
