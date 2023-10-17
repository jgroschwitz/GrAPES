from util import load_corpus_with_alignments, LEAMR_ALIGNMENTS, LEAMR_LDC_TEST_AMRS, \
    amr_utils_graph_to_penman_graph_with_all_explicit_names, get_aligned_tokens_for_amrutils_node_name
from vulcan_pickle_builder import VulcanPickleBuilder


def main():
    test_amrs, test_alignments = load_corpus_with_alignments(LEAMR_LDC_TEST_AMRS, LEAMR_ALIGNMENTS)

    vulcan_pickle_builder = VulcanPickleBuilder()

    for graph in test_amrs:
        penman_graph, node_map = amr_utils_graph_to_penman_graph_with_all_explicit_names(graph)
        alignments = test_alignments[graph.id]
        for edge in graph.edges:
            source_tokens = get_aligned_tokens_for_amrutils_node_name(edge[0], alignments)
            target_tokens = get_aligned_tokens_for_amrutils_node_name(edge[2], alignments)
            dist1 = max(source_tokens) - min(target_tokens)
            dist2 = max(target_tokens) - min(source_tokens)
            if dist1 > 20 or dist2 > 20:
                node_labels = [graph.nodes[x] for x in [edge[0], edge[2]]]
                if not any(x in ["multi-sentence", "and"] for x in node_labels):
                    vulcan_pickle_builder.add_graph(penman_graph)
                    vulcan_pickle_builder.add_graph_highlight([node_map[edge[0]], node_map[edge[2]]])
                    vulcan_pickle_builder.add_sent_highlight(list(set().union(source_tokens, target_tokens)))

    vulcan_pickle_builder.save_pickle("outputs/long_range_dependencies.pickle")


if __name__ == "__main__":
    main()
