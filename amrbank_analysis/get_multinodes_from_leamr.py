from collections import Counter
from random import Random

from util import load_corpus_with_alignments, LEAMR_ALIGNMENTS, LEAMR_LDC_TEST_AMRS, \
    amr_utils_graph_to_penman_graph_with_all_explicit_names, LEAMR_LDC_TRAIN_AMRS, \
    graph_string_from_connected_node_names
from vulcan_pickle_builder import VulcanPickleBuilder


def main():
    train_amrs, train_alignments = load_corpus_with_alignments(LEAMR_LDC_TRAIN_AMRS, LEAMR_ALIGNMENTS)
    test_amrs, test_alignments = load_corpus_with_alignments(LEAMR_LDC_TEST_AMRS, LEAMR_ALIGNMENTS)

    train_counter = get_multinode_constant_counts(test_alignments, train_amrs)
    test_counter = get_multinode_constant_counts(test_alignments, test_amrs)

    make_corpus_data(test_alignments, test_amrs)

    with open("outputs/mutinode_constants.txt", "w") as f:
        for label, count in test_counter.most_common():
            f.write(f"{label} {train_counter[label]} ({count})\n")


def get_multinode_constant_counts(test_alignments, train_amrs):
    train_counter = Counter()
    for amr_utils_graph in train_amrs:
        alignments = test_alignments[amr_utils_graph.id]
        for al in alignments:
            if len(al.nodes) > 1:
                node_labels = " ".join(
                    sorted([amr_utils_graph.nodes[nn] for nn in al.nodes if nn in amr_utils_graph.nodes]))
                if not "name" in node_labels and not "entity" in node_labels and not "quantity" in node_labels:
                    train_counter[node_labels] += 1
    return train_counter


def make_corpus_data(test_alignments, test_amrs):
    """
    Note that this shuffles the test_amrs in place
    :param test_alignments:
    :param test_amrs:
    :return:
    """
    r = Random(4942)  # random seed created with random.org
    r.shuffle(test_amrs)
    vulcan_pickle_builder = VulcanPickleBuilder()
    with open("../corpus/multinode_constants.tsv", "w") as f:
        for amr_utils_graph in test_amrs:
            alignments = test_alignments[amr_utils_graph.id]
            for al in alignments:
                if len(al.nodes) > 1:
                    node_labels = " ".join(
                        sorted([amr_utils_graph.nodes[nn] for nn in al.nodes if nn in amr_utils_graph.nodes]))
                    if "name" not in node_labels and "entity" not in node_labels and "quantity" not in node_labels:
                        graph, id_map = amr_utils_graph_to_penman_graph_with_all_explicit_names(amr_utils_graph)
                        penman_nns = [id_map[nn] for nn in al.nodes if nn in id_map]
                        al_graph_string = graph_string_from_connected_node_names(graph, penman_nns)
                        vulcan_pickle_builder.add_graph(graph)
                        vulcan_pickle_builder.add_graph_highlight([id_map[nn] for nn in al.nodes if nn in id_map])
                        f.write(f"{amr_utils_graph.id}\t{al_graph_string}\n")
    vulcan_pickle_builder.save_pickle("outputs/multinode_constants.pickle")


if __name__ == "__main__":
    main()
