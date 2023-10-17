import pickle
from typing import List


class VulcanPickleBuilder:

    def __init__(self):
        self.vulcan_graph_dict = {"type": "data", "name": "graphs", "format": "graph", "instances": []}
        self.vulcan_sent_dict = {"type": "data", "name": "sentences", "format": "tokenized_string", "instances": []}
        self.vulcan_id_dict = {"type": "data", "name": "ids", "format": "tokenized_string", "instances": []}

    def add_graph(self, penman_graph, add_sent=True, add_id=True):
        self.vulcan_graph_dict["instances"].append(penman_graph)
        if add_sent:
            self.add_sent_from_penman_graph(penman_graph)
        if add_id:
            self.vulcan_id_dict["instances"].append([penman_graph.metadata["id"]])

    def add_sent(self, sent):
        self.vulcan_sent_dict["instances"].append(sent)

    def add_sent_from_penman_graph(self, penman_graph):
        self.add_sent(penman_graph.metadata["snt"].split(" "))

    def add_id(self, graph_id):
        self.vulcan_id_dict["instances"].append([graph_id])

    def add_graph_highlight(self, highlight: List[str]):
        if "highlights" not in self.vulcan_graph_dict:
            self.vulcan_graph_dict["highlights"] = []
        self.vulcan_graph_dict["highlights"].append(highlight)

    def add_sent_highlight(self, highlight):
        if "highlights" not in self.vulcan_sent_dict:
            self.vulcan_sent_dict["highlights"] = []
        self.vulcan_sent_dict["highlights"].append(highlight)

    def save_pickle(self, path):
        pickle.dump([self.vulcan_graph_dict, self.vulcan_sent_dict, self.vulcan_id_dict], open(path, "wb"))
