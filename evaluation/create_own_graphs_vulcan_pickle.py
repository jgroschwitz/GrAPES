import pickle
import sys
from typing import List
from penman import load

from evaluation.file_utils import load_corpus_from_folder, read_tsv_with_comments


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
    To run this, run it from the main folder of the project and prepend "PYTHONPATH=./" to the command. The arguments
    are (1) the parser name, such as amrbart or amparser, and (2) the prefix of the tsv file, such as
     "ellipsis_filtered".

    Example:
    PYTHONPATH=./ python3 evaluation/create_own_graphs_vulcan_pickle.py amrbart bought_for

    This will then create a pickle file in "error_analysis/visual_inspection" called "amrbart-ellipsis_filtered.pickle",
    which can be loaded in Vulcan (using the main branch).
    :param args:
    :return:
    """
    parser_name = args[1]
    task_name = args[2]
    root_dir = "./"
    predicted_graphs = load(f"{root_dir}/{parser_name}-output/{task_name}.txt")
    gold_amrs = load(f"{root_dir}/corpus/{task_name}.txt")

    vulcan_pickle_builder = VulcanPickleBuilderOwnGraphComparison()
    for gold_graph, predicted_graph in zip(gold_amrs, predicted_graphs):
        vulcan_pickle_builder.add_gold_graph(gold_graph)
        vulcan_pickle_builder.add_predicted_graph(predicted_graph)

    vulcan_pickle_builder.save_pickle(f"{root_dir}/error_analysis/visual_inspection/{parser_name}-{task_name}.pkl")


if __name__ == "__main__":
    main(sys.argv)
