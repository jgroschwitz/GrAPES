import pickle
import sys
from typing import List
from penman import load


class VulcanPickleBuilderOwnGraphComparison:

    def __init__(self):
        self.vulcan_gold_graph_dict = {"type": "data", "name": "gold graph", "format": "graph", "instances": []}
        self.vulcan_predicted_graph_dict = {"type": "data", "name": "predicted graph", "format": "graph", "instances": []}
        self.vulcan_sent_dict = {"type": "data", "name": "sentence", "format": "tokenized_string", "instances": []}

    def add_gold_graph(self, penman_graph, add_sent=True):
        self.vulcan_gold_graph_dict["instances"].append(penman_graph)
        if add_sent:
            self.add_sent_from_penman_graph(penman_graph)

    def add_predicted_graph(self, penman_graph):
        self.vulcan_predicted_graph_dict["instances"].append(penman_graph)

    def add_sent_from_penman_graph(self, penman_graph):
        self.add_sent(penman_graph.metadata["snt"].split(" "))

    def add_sent(self, sent):
        self.vulcan_sent_dict["instances"].append(sent)

    def save_pickle(self, path):
        pickle.dump([self.vulcan_gold_graph_dict, self.vulcan_predicted_graph_dict,
                     self.vulcan_sent_dict], open(path, "wb"))


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
    main(sys.argv)
