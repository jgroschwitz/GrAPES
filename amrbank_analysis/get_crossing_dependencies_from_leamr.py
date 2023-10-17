from util import load_corpus_with_alignments, LEAMR_ALIGNMENTS, LEAMR_LDC_TEST_AMRS, \
    amr_utils_graph_to_penman_graph_with_all_explicit_names, get_aligned_tokens_for_amrutils_node_name
from vulcan_pickle_builder import VulcanPickleBuilder


def main():
    test_amrs, test_alignments = load_corpus_with_alignments(LEAMR_LDC_TEST_AMRS, LEAMR_ALIGNMENTS)

    vulcan_pickle_builder = VulcanPickleBuilder()

    i = 0
    for graph in test_amrs:
        penman_graph, node_map = amr_utils_graph_to_penman_graph_with_all_explicit_names(graph)
        alignments = test_alignments[graph.id]
        seen_graph_edges = set()
        for edge in graph.edges:
            seen_graph_edges.add(edge)
            if not any(edge[0] in al.nodes and edge[2] in al.nodes for al in alignments):
                source_tokens = get_aligned_tokens_for_amrutils_node_name(edge[0], alignments)
                target_tokens = get_aligned_tokens_for_amrutils_node_name(edge[2], alignments)
                all_edge_tokens = set().union(source_tokens, target_tokens)
                whole_edge_span = range(min(all_edge_tokens), max(all_edge_tokens) + 1)
                for edge2 in graph.edges:
                    if edge2 not in seen_graph_edges:
                        if not any(edge2[0] in al.nodes and edge2[2] in al.nodes for al in alignments):
                            source_tokens2 = get_aligned_tokens_for_amrutils_node_name(edge2[0], alignments)
                            target_tokens2 = get_aligned_tokens_for_amrutils_node_name(edge2[2], alignments)
                            all_edge_tokens2 = set().union(source_tokens2, target_tokens2)
                            nodes = [node_map[edge[0]], node_map[edge[2]], node_map[edge2[0]], node_map[edge2[2]]]
                            if any(x not in whole_edge_span for x in all_edge_tokens2) and\
                                    any(x in whole_edge_span for x in all_edge_tokens2) and\
                                    not any(x in all_edge_tokens for x in all_edge_tokens2):
                                if not any(inst.target == "multi-sentence" for inst in penman_graph.instances() if inst.source in nodes):
                                    if i < 500:
                                        vulcan_pickle_builder.add_graph(penman_graph)
                                        vulcan_pickle_builder.add_graph_highlight([node_map[edge[0]], node_map[edge[2]],
                                                                                   node_map[edge2[0]], node_map[edge2[2]]])
                                        vulcan_pickle_builder.add_sent_highlight(list(set().union(source_tokens, target_tokens,
                                                                                                  source_tokens2, target_tokens2)))
                                    i += 1

    print(i)
    vulcan_pickle_builder.save_pickle("outputs/crossing_dependencies.pickle")


if __name__ == "__main__":
    main()
