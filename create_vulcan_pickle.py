import argparse
import os
import pickle
import re
from penman import load, dump
from amconll import parse_amconll, Entry, write_conll, AMSentence
from vulcan.pickle_builder.pickle_builder import PickleBuilder

from evaluation.file_utils import read_edge_tsv, read_tsv_with_comments






class VulcanPickleBuilderOwnGraphComparison:

    def __init__(self):
        self.vulcan_gold_graph_dict = {"type": "data", "name": "gold graph", "format": "graph", "instances": []}
        self.vulcan_predicted_graph_dict = {"type": "data", "name": "predicted graph", "format": "graph", "instances": []}
        self.vulcan_sent_dict = {"type": "data", "name": "sentence", "format": "tokenized_string", "instances": []}
        self.vulcan_amconll_dict = {"type": "data", "name": "amconll", "format": "object_table", "instances": []}

    def add_gold_graph(self, penman_graph, add_sent=True):
        self.vulcan_gold_graph_dict["instances"].append(penman_graph)
        if add_sent:
            self.add_sent_from_penman_graph(penman_graph)

    def add_predicted_graph(self, penman_graph):
        self.vulcan_predicted_graph_dict["instances"].append(penman_graph)

    def add_sent_from_penman_graph(self, penman_graph):
        self.add_sent(penman_graph.metadata["snt"].split(" "))

    def add_sent(self, sent):
        print("adding sent")
        self.vulcan_sent_dict["instances"].append(sent)

    def add_am_tree(self, am_tree):
        self.vulcan_amconll_dict["instances"].append(am_tree)

    def save_pickle(self, path):
        """
        Write a pickle with everything that's non-empty
        Args:
            path: path to save pickle to
        """
        all_elements = [self.vulcan_gold_graph_dict, self.vulcan_predicted_graph_dict,
                         self.vulcan_sent_dict, self.vulcan_amconll_dict]
        included = [element for element in all_elements if len(element["instances"]) > 0]
        with open(path, "wb") as f:
            pickle.dump(included, f)


def write_amr_corpora_for_testset_subcategory(prediction_path, gold_path, subcategory_tsv,
                                             subcategory_name=None, out_path="error_analysis", graph_id_column=0):
    """
    Using the graph IDs in the TSV, extract the relevant gold and predicted graphs and write them to AMR corpus files.
    Args:
        prediction_path: path to the AMR corpus prediction file.
        gold_path: path to the AMR corpus gold file.
        subcategory_tsv: path to the TSV for the subcategory.
        subcategory_name: name of the subcategory for output file naming (default the TSV name)
        out_path: path to DIRECTORY to write the new corpora to. Default error_analysis.
        graph_id_column: if the graph ID is not in column 0, this tells us where to find the ID in the TSV
    """
    predicted_graphs = load(prediction_path)
    gold_graphs = load(gold_path)
    if subcategory_name is None:
        # use the TSV filename
        subcategory_name = os.path.basename(subcategory_tsv)[:-4]

    # write here
    out_gold_path = f"{out_path}/{subcategory_name}_gold.txt"
    out_predicted_path = f"{out_path}/{subcategory_name}_pred.txt"

    # the IDs of the graphs we actually want
    ids = get_graph_ids_from_tsv(graph_id_column, subcategory_tsv)

    predictions = []
    golds = []
    for i in range(len(gold_graphs)):
        gold_graph = gold_graphs[i]
        graph_id = gold_graph.metadata["id"]
        if graph_id in ids:
            sent = gold_graph.metadata["snt"]
            predicted_graph = predicted_graphs[i]
            # add the metadata from the gold graph
            predicted_graph.metadata["snt"] = sent
            predicted_graph.metadata["id"] = graph_id
            golds.append(gold_graph)
            predictions.append(predicted_graph)
    dump(golds, out_gold_path)
    dump(predictions, out_predicted_path)
    # return info so we can access outputs easily from build_pickle_for_testset_subcategory
    return out_predicted_path, out_gold_path, ids, subcategory_name


def build_pickle_for_testset_subcategory(prediction_path, gold_path, subcategory_tsv, pickle_directory,
                                         graph_id_column=0, subcategory_name=None):

    # write intermediate files
    out_predicted_path, out_gold_path, ids, subcategory_name = write_amr_corpora_for_testset_subcategory(prediction_path,
                                                                                  gold_path,
                                                                                  subcategory_tsv,
                                                                                  subcategory_name,
                                                                                  pickle_directory,
                                                                                  graph_id_column)
    # read in subcorpora
    predicted_graphs = load(out_predicted_path)
    gold_graphs = load(out_gold_path)

    vulcan_pickle_builder = VulcanPickleBuilderOwnGraphComparison()
    for i in range(len(gold_graphs)):
        gold_graph = gold_graphs[i]
        include_sentence = True
        predicted_graph = predicted_graphs[i]
        vulcan_pickle_builder.add_gold_graph(gold_graph, include_sentence)
        vulcan_pickle_builder.add_predicted_graph(predicted_graph)


    pickle_path = f"{pickle_directory}/{subcategory_name}.pickle"
    vulcan_pickle_builder.save_pickle(pickle_path)
    print(f"Wrote pickle to {pickle_path}")
    print(f"Wrote AMR corpora to {out_gold_path} and {out_predicted_path}")



def get_graph_ids_from_tsv(graph_id_column, subcategory_tsv):
    ids = []
    with open(subcategory_tsv, "r", encoding="utf8") as f:
        csvreader = read_tsv_with_comments(f)
        for row in csvreader:
            graph_id = row[graph_id_column]
            ids.append(graph_id)
    return ids




def main(args):
    """
    Creates a vulcan-readable pickle of a GrAPES category. Works for any pair of predicted and gold files with the graphs in the same order.
    
    Arguments to this script are:
    	1. Path to the predictions file
    	2. Path to the gold file
    	3. Output path for the pickle

    :param args: list of strings (from sys.argv)
    """
    parser_output_path = args[1]
    gold_path = args[2]
    pickle_path = args[3]
    predicted_graphs = load(parser_output_path)
    gold_amrs = load(gold_path)
    
    vulcan_pickle_builder = VulcanPickleBuilderOwnGraphComparison()
    for gold_graph, predicted_graph in zip(gold_amrs, predicted_graphs):
        vulcan_pickle_builder.add_gold_graph(gold_graph)
        vulcan_pickle_builder.add_predicted_graph(predicted_graph)

    vulcan_pickle_builder.save_pickle(pickle_path)


if __name__ == "__main__":
    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("-p", "--predictions_path", help="Path to the predictions file", required=True)
    command_line_parser.add_argument("-g", "--gold_path", help="Path to the gold file", required=True)
    command_line_parser.add_argument("-o", "--output_path", help="Path to the output folder", default="error_analysis")
    command_line_parser.add_argument("-s", "--subcategory_name", help="Name of the subcategory (default all)", default=None)
    command_line_parser.add_argument("-e", "--error_analysis_pickle_path", help="Path to the error analysis pickle file", required=True)

    args = command_line_parser.parse_args()
    if args.tsv is not None:
        build_pickle_for_testset_subcategory(args.predictions_path, args.gold_path, args.tsv,
                                             args.output_path, args.id_column_number, args.subcategory_name)
        # write_amr_corpora_for_testset_subcategory(args.predictions_path, args.gold_path, args.tsv,  out_path=args.output_path)
    else:
        main([None, args.predictions_path, args.gold_path, args.output_path])