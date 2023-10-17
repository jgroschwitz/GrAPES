import pickle
import sys
from typing import List
from penman import load

from evaluation.file_utils import load_corpus_from_folder, read_tsv_with_comments
from evaluation.util import strip_sense

class VulcanPickleBuilderTSVComparison:

    def __init__(self):
        self.vulcan_gold_graph_dict = {"type": "data", "name": "gold graph", "format": "graph", "instances": []}
        self.vulcan_predicted_graph_dict = {"type": "data", "name": "predicted graph", "format": "graph", "instances": []}
        self.vulcan_sent_dict = {"type": "data", "name": "sentence", "format": "tokenized_string", "instances": []}
        self.vulcan_tsv_dict = {"type": "data", "name": "TSV reference", "format": "tokenized_string", "instances": []}

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

    def add_tsv_row(self, tsv_row: List[str]):
        self.vulcan_tsv_dict["instances"].append(tsv_row)

    def save_pickle(self, path):
        pickle.dump([self.vulcan_gold_graph_dict, self.vulcan_predicted_graph_dict,
                     self.vulcan_sent_dict, self.vulcan_tsv_dict], open(path, "wb"))


def main(args):
    """
    To run this, run it from the main folder of the project and prepend "PYTHONPATH=./" to the command. The arguments
    are (1) the parser name, such as amrbart or amparser, and (2) the prefix of the tsv file, such as
     "ellipsis_filtered".

    Example:
    PYTHONPATH=./ python3 evaluation/testset/create_testset_vulcan_pickle.py amrbart ellipsis_filtered

    This will then create a pickle file in "error_analysis/visual_inspection" called "amrbart-ellipsis_filtered.pickle",
    which can be loaded in Vulcan (using the main branch).
    :param args:
    :return:
    """
    parser_name = args[1]
    task_name = args[2]
    root_dir = "./"
    predicted_graphs = load(f"{root_dir}/{parser_name}-output/testset.txt")
    gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")

    id2rows = dict()
    with open(f"{root_dir}/corpus/{task_name}.tsv") as f:
        tsv_file = read_tsv_with_comments(f)
        for row in tsv_file:
            rows = id2rows.setdefault(row[0], [])
            rows.append(row)

    vulcan_pickle_builder = VulcanPickleBuilderTSVComparison()
    for gold_graph, predicted_graph in zip(gold_amrs, predicted_graphs):
        id_here = gold_graph.metadata["id"]
        if id_here in id2rows:
            for row in id2rows[id_here]:
                vulcan_pickle_builder.add_gold_graph(gold_graph)
                vulcan_pickle_builder.add_predicted_graph(predicted_graph)
                vulcan_pickle_builder.add_tsv_row(row)

    vulcan_pickle_builder.save_pickle(f"{root_dir}/error_analysis/visual_inspection/{parser_name}-{task_name}.pkl")

def main_but_only_sense_errors(args):
    """
    To run this, run it from the main folder of the project and prepend "PYTHONPATH=./" to the command. The arguments
    are (1) the parser name, such as amrbart or amparser, and (2) the prefix of the tsv file, such as
     "ellipsis_filtered".

    Example:
    PYTHONPATH=./ python3 evaluation/testset/create_testset_vulcan_pickle.py amrbart ellipsis_filtered

    This will then create a pickle file in "error_analysis/visual_inspection" called "amrbart-ellipsis_filtered.pickle",
    which can be loaded in Vulcan (using the main branch).
    :param args:
    :return:
    """
    parser_name = args[1]
    task_name = args[2]
    root_dir = "./"
    predicted_graphs = load(f"{root_dir}/{parser_name}-output/testset.txt")
    gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")

    id2rows = dict()
    with open(f"{root_dir}/corpus/{task_name}.tsv") as f:
        tsv_file = read_tsv_with_comments(f)
        for row in tsv_file:
            rows = id2rows.setdefault(row[0], [])
            rows.append(row)

    vulcan_pickle_builder = VulcanPickleBuilderTSVComparison()
    for gold_graph, predicted_graph in zip(gold_amrs, predicted_graphs):
        id_here = gold_graph.metadata["id"]
        predicted_node_labels = [inst.target for inst in predicted_graph.instances()]
        predicted_lemmas = [strip_sense(inst.target) for inst in predicted_graph.instances()]
        if id_here in id2rows:
            for row in id2rows[id_here]:
                if (row[1] not in predicted_node_labels) and strip_sense(row[1]) in predicted_lemmas:
                    vulcan_pickle_builder.add_gold_graph(gold_graph)
                    vulcan_pickle_builder.add_predicted_graph(predicted_graph)
                    vulcan_pickle_builder.add_tsv_row(row)

    vulcan_pickle_builder.save_pickle(f"{root_dir}/error_analysis/visual_inspection/{parser_name}-{task_name}.pkl")



if __name__ == "__main__":
    main(sys.argv)
    # main_but_only_sense_errors(sys.argv)