import re
from collections import Counter
from find_rare_node_labels import load_corpus_from_folder
from util import get_node_by_name
from vulcan_pickle_builder import VulcanPickleBuilder


def main():
    test_corpus = load_corpus_from_folder("../../../data/Edinburgh/amr3.0/data/amrs/split/test/")
    print(len(test_corpus))

    case_counter = Counter()

    vulcan_pickle_builder = VulcanPickleBuilder()

    for graph in test_corpus:
        for attribute in graph.attributes(role=":wiki"):
            if attribute.target == "-":
                case_counter["no wiki"] += 1
                vulcan_pickle_builder.add_graph(graph)
                vulcan_pickle_builder.add_graph_highlight([attribute.source])
            else:
                case_counter["yes wiki"] += 1

    print(case_counter)
    vulcan_pickle_builder.save_pickle("outputs/wiki_required.pickle")


if __name__ == "__main__":
    main()
