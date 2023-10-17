from util import load_corpus_with_alignments, LEAMR_ALIGNMENTS, LEAMR_LDC_TEST_AMRS, \
    amr_utils_graph_to_penman_graph_with_all_explicit_names, get_node_by_name
from vulcan_pickle_builder import VulcanPickleBuilder


def main():
    test_amrs, test_alignments = load_corpus_with_alignments(LEAMR_LDC_TEST_AMRS, LEAMR_ALIGNMENTS)

    vulcan_pickle_builder = VulcanPickleBuilder()
    vulcan_pickle_builder_constrained = VulcanPickleBuilder()
    with open("../corpus/ellipsis.tsv", "w") as f:
        for amr_utils_graph in test_amrs:
            alignments = test_alignments[amr_utils_graph.id]
            seen_spans_to_nodes = dict()
            ellipsis_tokens = set()
            ellipsis_nodes = set()
            graph, node_map = amr_utils_graph_to_penman_graph_with_all_explicit_names(amr_utils_graph)
            for al in alignments:
                if hashable_span(al.tokens) in seen_spans_to_nodes:
                    if node_label_set(al.nodes, node_map, graph) == \
                            node_label_set(seen_spans_to_nodes[hashable_span(al.tokens)], node_map, graph):
                        ellipsis_tokens.update(al.tokens)
                        ellipsis_nodes.update([node_map[n] for n in al.nodes])
                        ellipsis_nodes.update([node_map[n] for n in seen_spans_to_nodes[hashable_span(al.tokens)]])
                seen_spans_to_nodes[hashable_span(al.tokens)] = al.nodes
            for ellipsis_node_label in set([get_node_by_name(nn, graph).target for nn in ellipsis_nodes]):
                if len([instance for instance in graph.instances() if instance.target == ellipsis_node_label]) == 2:
                    f.write(f"{graph.metadata['id']}\t{ellipsis_node_label}\n")
                    vulcan_pickle_builder_constrained.add_graph(graph)
                    vulcan_pickle_builder_constrained.add_graph_highlight(
                        [instance.source for instance in graph.instances() if instance.target == ellipsis_node_label])
                    vulcan_pickle_builder_constrained.add_sent_highlight(list(ellipsis_tokens))
            if len(ellipsis_nodes) > 0:
                vulcan_pickle_builder.add_graph(graph)
                vulcan_pickle_builder.add_graph_highlight(list(ellipsis_nodes))
                vulcan_pickle_builder.add_sent_highlight(list(ellipsis_tokens))

    vulcan_pickle_builder_constrained.save_pickle("outputs/ellipsis_constrained.pickle")
    vulcan_pickle_builder.save_pickle(f"outputs/ellipsis.pickle")


def hashable_span(span):
    return tuple(span)


def node_label_set(node_names, node_map, graph):
    return set([get_node_by_name(node_map[n], graph).target for n in node_names])


if __name__ == "__main__":
    main()
