import argparse
import os
import pickle
import re
from typing import Tuple, Callable, List

import penman
from penman import load, dump, Graph
from amconll import parse_amconll, Entry, write_conll, AMSentence
from vulcan.data_handling.format_names import FORMAT_NAME_TOKENIZED_STRING, FORMAT_NAME_GRAPH, FORMAT_NAME_STRING
from vulcan.pickle_builder.pickle_builder import PickleBuilder

from evaluate_single_category import get_gold_path_based_on_info
from evaluation.file_utils import read_edge_tsv, read_tsv_with_comments
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo


# class VulcanPickleBuilderOwnGraphComparison:
#
#     def __init__(self):
#         self.vulcan_gold_graph_dict = {"type": "data", "name": "gold graph", "format": "graph", "instances": []}
#         self.vulcan_predicted_graph_dict = {"type": "data", "name": "predicted graph", "format": "graph", "instances": []}
#         self.vulcan_sent_dict = {"type": "data", "name": "sentence", "format": "tokenized_string", "instances": []}
#         self.vulcan_amconll_dict = {"type": "data", "name": "amconll", "format": "object_table", "instances": []}
#
#     def add_gold_graph(self, penman_graph, add_sent=True):
#         self.vulcan_gold_graph_dict["instances"].append(penman_graph)
#         if add_sent:
#             self.add_sent_from_penman_graph(penman_graph)
#
#     def add_predicted_graph(self, penman_graph):
#         self.vulcan_predicted_graph_dict["instances"].append(penman_graph)
#
#     def add_sent_from_penman_graph(self, penman_graph):
#         self.add_sent(penman_graph.metadata["snt"].split(" "))
#
#     def add_sent(self, sent):
#         print("adding sent")
#         self.vulcan_sent_dict["instances"].append(sent)
#
#     def add_am_tree(self, am_tree):
#         self.vulcan_amconll_dict["instances"].append(am_tree)
#
#     def save_pickle(self, path):
#         """
#         Write a pickle with everything that's non-empty
#         Args:
#             path: path to save pickle to
#         """
#         all_elements = [self.vulcan_gold_graph_dict, self.vulcan_predicted_graph_dict,
#                          self.vulcan_sent_dict, self.vulcan_amconll_dict]
#         included = [element for element in all_elements if len(element["instances"]) > 0]
#         with open(path, "wb") as f:
#             pickle.dump(included, f)
#
#
# def write_amr_corpora_for_testset_subcategory(prediction_path, gold_path,
#                                              subcategory_name=None, out_path="error_analysis"):
#     """
#     Using the graph IDs in the TSV, extract the relevant gold and predicted graphs and write them to AMR corpus files.
#     Args:
#         prediction_path: path to the AMR corpus prediction file.
#         gold_path: path to the AMR corpus gold file.
#         subcategory_tsv: path to the TSV for the subcategory.
#         subcategory_name: name of the subcategory for output file naming (default the TSV name)
#         out_path: path to DIRECTORY to write the new corpora to. Default error_analysis.
#         graph_id_column: if the graph ID is not in column 0, this tells us where to find the ID in the TSV
#     """
#     predicted_graphs = load(prediction_path)
#     gold_graphs = load(gold_path)
#
#     # write here
#     out_gold_path = f"{out_path}/{subcategory_name}_gold.txt"
#     out_predicted_path = f"{out_path}/{subcategory_name}_pred.txt"
#
#     # the IDs of the graphs we actually want
#     ids = get_graph_ids_from_tsv(graph_id_column, subcategory_tsv)
#
#     predictions = []
#     golds = []
#     for i in range(len(gold_graphs)):
#         gold_graph = gold_graphs[i]
#         graph_id = gold_graph.metadata["id"]
#         if graph_id in ids:
#             sent = gold_graph.metadata["snt"]
#             predicted_graph = predicted_graphs[i]
#             # add the metadata from the gold graph
#             predicted_graph.metadata["snt"] = sent
#             predicted_graph.metadata["id"] = graph_id
#             golds.append(gold_graph)
#             predictions.append(predicted_graph)
#     dump(golds, out_gold_path)
#     dump(predictions, out_predicted_path)
#     # return info so we can access outputs easily from build_pickle_for_testset_subcategory
#     return out_predicted_path, out_gold_path, ids, subcategory_name


def build_pickle_for_testset_subcategory(prediction_path, gold_path, pickle_directory, subcategory_name=None):
    x: Tuple[Callable, SubcategoryMetadata] = category_name_to_set_class_and_metadata[subcategory_name]
    eval_class, info = x

    instance_info = EvaluationInstanceInfo(
        absolute_path_to_predictions_file=prediction_path,
        absolute_path_to_gold_file=gold_path,
    )

    print(info.display_name)
    gold_path, predictions_path, use_subcorpus = get_gold_path_based_on_info(gold_path, info, instance_info)

    gold_amrs = penman.load(gold_path)
    predicted_amrs = penman.load(predictions_path)

    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)
    return dummy_evaluator.gold_amrs, dummy_evaluator.predicted_amrs


def create_pickle(gold_graphs: List[Graph], predicted_graphs: List[Graph], path_to_pickle: str):
    """
    Read in AM Parser output files and create a vulcan-readable pickle
    Pickle contains AM dependency tree, including graph constants, and gold and predicted graphs.
    Args:
        gold_graphs: the gold graphs, read in and filtered to contain exactly the relevant ones
        predicted_graphs: the predictions, same order as gold
        path_to_pickle: output path to write vulcan pickle to
    """

    # initialise the pickle builder with the appropriate fields and their data types
    # The sentence is a table because it will stack the supertags on the words for each word
    pickle_builder = PickleBuilder({"Gold graph": FORMAT_NAME_GRAPH, "Predicted graph": FORMAT_NAME_GRAPH,
                                    "Sentence": FORMAT_NAME_STRING, "ID": FORMAT_NAME_STRING})

    # everything is in the same order, so we can zip the lists to get all info for each corpus entry
    for gold_amr, predicted_amr in zip(gold_graphs, predicted_graphs):

        # use the exact same field names as used when the pickle builder was initialised.
        pickle_builder.add_instances_by_name({"Gold graph": gold_amr,
                                              "Predicted graph": predicted_amr,
                                              "Sentence": gold_amr.metadata["snt"],
                                              "ID": gold_amr.metadata["id"]})

    # write the pickle
    pickle_builder.write(path_to_pickle)


if __name__ == "__main__":
    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("-p", "--predictions_path", help="Path to the predictions file", required=True)
    command_line_parser.add_argument("-g", "--gold_path", help="Path to the gold file", required=True)
    command_line_parser.add_argument("-o", "--output_path", help="Path to the output folder", default="error_analysis")
    command_line_parser.add_argument("-s", "--subcategory_name", help="Name of the subcategory (default all)", default=None)
    command_line_parser.add_argument("-e", "--error_analysis_pickle_path", help="Path to the error analysis pickle file", required=False)

    args = command_line_parser.parse_args()
    golds, preds = build_pickle_for_testset_subcategory(args.predictions_path, args.gold_path,
                                             args.output_path, args.subcategory_name)

    create_pickle(golds, preds, args.output_path)