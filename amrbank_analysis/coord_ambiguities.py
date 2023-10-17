import re
from collections import Counter
from find_rare_node_labels import load_corpus_from_folder
from util import get_node_by_name, load_corpus_with_alignments, LEAMR_LDC_TEST_AMRS, LEAMR_ALIGNMENTS, \
    amr_utils_graph_to_penman_graph_with_all_explicit_names
from vulcan_pickle_builder import VulcanPickleBuilder


def main():
    test_amrs, test_alignments = load_corpus_with_alignments(LEAMR_LDC_TEST_AMRS, LEAMR_ALIGNMENTS)
    print(len(test_amrs))

    vulcan_pickle_builder = VulcanPickleBuilder()
    case_counter = Counter()
    for amr_utils_graph in test_amrs:
        alignments = test_alignments[amr_utils_graph.id]
        graph, node_map = amr_utils_graph_to_penman_graph_with_all_explicit_names(amr_utils_graph)
        for nn in amr_utils_graph.nodes:
            if amr_utils_graph.nodes[nn] in ["and", "or"]:
                modifiers_high = []
                modifiers_low = []
                children = []
                min_alignment = 100000
                max_alignment = -1
                for src, role, tgt in amr_utils_graph.edges:
                    if src == nn and role == ":mod":
                        modifiers_high.append(tgt)
                    if src == nn and role.startswith(":op"):
                        children.append(tgt)
                print("children")
                print([amr_utils_graph.nodes[x] for x in children])
                for child_nn in children:
                    for al in alignments:
                        if child_nn in al.nodes:
                            min_alignment = min(min_alignment, min(al.tokens))
                            max_alignment = max(max_alignment, max(al.tokens))
                    for src, role, tgt in amr_utils_graph.edges:
                        if src == child_nn and not (role.startswith(":ARG") or role.startswith(":op")):
                            modifiers_low.append(tgt)
                print("modifiers_low")
                print([amr_utils_graph.nodes[x] for x in modifiers_low])
                for al in alignments:
                    if max(al.tokens) < min_alignment:
                        if any(x in modifiers_high for x in al.nodes):
                            case_counter["left, high"] += 1
                            vulcan_pickle_builder.add_graph(graph)
                            vulcan_pickle_builder.add_graph_highlight([node_map[x] for x in al.nodes])
                            vulcan_pickle_builder.add_sent_highlight([x for x in al.tokens])
                        elif any(x in modifiers_low for x in al.nodes):
                            case_counter["left, low"] += 1
                            vulcan_pickle_builder.add_graph(graph)
                            vulcan_pickle_builder.add_graph_highlight([node_map[x] for x in al.nodes])
                            vulcan_pickle_builder.add_sent_highlight([x for x in al.tokens])
                    elif min(al.tokens) > max_alignment:
                        if any(x in modifiers_high for x in al.nodes):
                            case_counter["right, high"] += 1
                            vulcan_pickle_builder.add_graph(graph)
                            vulcan_pickle_builder.add_graph_highlight([node_map[x] for x in al.nodes])
                            vulcan_pickle_builder.add_sent_highlight([x for x in al.tokens])
                        elif any(x in modifiers_low for x in al.nodes):
                            case_counter["right, low"] += 1
                            vulcan_pickle_builder.add_graph(graph)
                            vulcan_pickle_builder.add_graph_highlight([node_map[x] for x in al.nodes])
                            vulcan_pickle_builder.add_sent_highlight([x for x in al.tokens])

    vulcan_pickle_builder.save_pickle("outputs/coord_ambiguities.pickle")

    with open("outputs/coord_ambiguities.tsv", "w") as f:
        for case, count in case_counter.most_common():
            f.write(f"{case}\t{count}\n")


if __name__ == '__main__':
    main()
