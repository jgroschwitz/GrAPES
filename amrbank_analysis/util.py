import os
import sys
from typing import List, Set

import penman
from penman import Graph, load
from penman.graph import CONCEPT_ROLE
from amr_utils.amr_readers import AMR_Reader


LEAMR_LDC_TEST_AMRS = "../external_resources/leamr-data-release/amrs/ldc_test.txt"
LEAMR_LDC_TRAIN_AMRS = "../external_resources/leamr-data-release/amrs/ldc_train.txt"
LEAMR_ALIGNMENTS = "../external_resources/leamr-data-release/alignments/ldc+little_prince.subgraph_alignments.json"


def turn_attributes_into_nodes(graph: Graph):
    running_node_label_id = 1
    for attribute in graph.attributes():
        node_label = "x" + str(running_node_label_id) if running_node_label_id > 1 else "x"
        running_node_label_id += 1
        new_instance = penman.Triple(node_label, CONCEPT_ROLE, attribute.target.replace('"', ''))
        graph.triples.append(new_instance)
        new_edge = penman.Triple(attribute.source, attribute.role, node_label)
        graph.triples.append(new_edge)
        graph.triples.remove(attribute)


def get_node_by_name(node_name: str, graph: Graph):
    return [n for n in graph.instances() if n.source == node_name][0]


def load_corpus_from_folder(folder_path: str):
    corpus = []
    for file in sorted(os.listdir(folder_path)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            corpus.extend(load(folder_path + filename))
    return corpus


def get_name(name_node_name, graph):
    """
    Gets the attributes of the given node in the graph, returning them joined into a string.
        Removes quotation marks.
        Intended to be used with a "name" node and its opi attributes, so the name comes out in order
    :param name_node_name:
    :param graph: penman graph
    :return: " ".join of the attributes in order
    """
    ret = []
    for attribute in sorted(graph.attributes(source=name_node_name), key=lambda x: x.role):
        ret.append(attribute.target.replace('"', ''))
    return " ".join(ret)


def load_corpus_with_alignments(amr_file_path, alignment_file_path):
    reader = AMR_Reader()  # MyAMRReader()
    amrs = reader.load(amr_file_path, remove_wiki=True)
    alignment_corpus = reader.load_alignments_from_json(alignment_file_path, amrs)
    print(len(amrs))
    # print(amrs[0])
    print(len(alignment_corpus))
    # print(alignment_corpus[amrs[0].id])
    return amrs, alignment_corpus


def amr_utils_graph_to_penman_graph_with_all_explicit_names(amr):
    """
    Adapted (just slightly) from amr-utils method graph_string
    :param amr:
    :return:
    """
    amr_string = f'[[{amr.root}]]'
    new_ids = {}
    for n in amr.nodes:
        new_id = amr.nodes[n][0] if amr.nodes[n] else 'x'
        if new_id.isalpha() and new_id.islower():
            if new_id in new_ids.values():
                j = 2
                while f'{new_id}{j}' in new_ids.values():
                    j += 1
                new_id = f'{new_id}{j}'
        else:
            j = 0
            while f'x{j}' in new_ids.values():
                j += 1
            new_id = f'x{j}'
        new_ids[n] = new_id
    depth = 1
    nodes = {amr.root}
    completed = set()
    while '[[' in amr_string:
        tab = '\t' * depth
        for n in nodes.copy():
            id = new_ids[n] if n in new_ids else 'x91'
            concept = amr.nodes[n] if n in new_ids and amr.nodes[n] else 'None'
            edges = sorted([e for e in amr.edges if e[0] == n], key=lambda x: x[1])
            targets = set(t for s, r, t in edges)
            edge_strings = [f'{r} [[{t}]]' for s, r, t in edges]
            # edge_strings = [f'{r} [[{t}]]' for s, r, t in edges if (t not in completed)]
            # edge_strings += [f'{r} {new_ids[t]}' for s, r, t in edges if (t in completed)]
            # edges = [f'{r} [[{t}]]' for s, r, t in edges]
            children = f'\n{tab}'.join(edge_strings)
            if children:
                children = f'\n{tab}' + children
            if n not in completed:
                # here is my change to just always print the node name
                amr_string = amr_string.replace(f'[[{n}]]', f'({id}/{concept}{children})', 1)
                # if (concept[0].isalpha() and concept not in ['imperative', 'expressive', 'interrogative']) or targets:
                #     amr_string = amr_string.replace(f'[[{n}]]', f'({id}/{concept}{children})', 1)
                # else:
                #     amr_string = amr_string.replace(f'[[{n}]]', f'{concept}')
                completed.add(n)
            amr_string = amr_string.replace(f'[[{n}]]', f'{id}')
            nodes.remove(n)
            nodes.update(targets)
        depth += 1
    if len(completed) < len(amr.nodes):
        missing_nodes = [n for n in amr.nodes if n not in completed]
        missing_edges = [(s, r, t) for s, r, t in amr.edges if s in missing_nodes or t in missing_nodes]
        missing_nodes= ', '.join(f'{n}/{amr.nodes[n]}' for n in missing_nodes)
        missing_edges = ', '.join(f'{s}/{amr.nodes[s]} {r} {t}/{amr.nodes[t]}' for s,r,t in missing_edges)
        print('[amr]', 'Failed to print AMR, '
              + str(len(completed)) + ' of ' + str(len(amr.nodes)) + ' nodes printed:\n '
              + str(amr.id) + ':\n'
              + amr_string + '\n'
              + 'Missing nodes: ' + missing_nodes +'\n'
              + 'Missing edges: ' + missing_edges +'\n',
              file=sys.stderr)
    if not amr_string.startswith('('):
        amr_string = '(' + amr_string + ')'
    if len(amr.nodes) == 0:
        amr_string = '(a/amr-empty)'

    penman_graph = penman.decode(amr_string)
    penman_graph.metadata['id'] = amr.id
    penman_graph.metadata['snt'] = " ".join(amr.tokens)

    return penman_graph, new_ids


def get_aligned_tokens_for_amrutils_node_name(node_name, alignments):
    return set().union(*[al.tokens for al in alignments if node_name in al.nodes])


def node_name_to_reference_graph_string(node_name, graph):
    """

    :param node_name:
    :param graph: penman library graph (based on triples)
    :return: A string representation of a subgraph of the given graph (modulo node names), containing the node
    specified by node_name, that hopefully contains enough unique information to find the node in the graph
    uniquely. Node_name is always the root of the reference graph.
    """
    name_edges = graph.edges(source=node_name, role=":name")
    if len(name_edges) == 0:
        return f"({node_name} / {get_node_by_name(node_name, graph).target})"
    else:
        name_node = name_edges[0].target
        return_string = f"({node_name} / {get_node_by_name(node_name, graph).target} :name ({name_node} / name"
        for attr in graph.attributes(source=name_node):
            attr_label = attr.target.replace("\"", "")
            return_string += f" {attr.role} \"{attr_label}\""
        return return_string + "))"


def graph_string_from_connected_node_names(penman_graph, node_names: List[str]):
    seen_node_names = set()
    expanded_graph = _expand_graph_with_node_name(penman_graph, node_names[0], node_names, seen_node_names)
    if len(seen_node_names) == len(node_names):
        return expanded_graph
    else:
        return None


def _expand_graph_with_node_name(penman_graph, current_node_name: str, node_names: List[str], seen_node_names: Set[str]):
    ret = f"({current_node_name} / {get_node_by_name(current_node_name, penman_graph).target}"
    seen_node_names.add(current_node_name)
    for edge in penman_graph.edges(source=current_node_name):
        if edge.target in node_names and edge.target not in seen_node_names:
            ret += f" {edge.role} "
            ret += _expand_graph_with_node_name(penman_graph, edge.target, node_names, seen_node_names)
    for edge in penman_graph.edges(target=current_node_name):
        if edge.source in node_names and edge.source not in seen_node_names:
            ret += f" {edge.role}-of "
            ret += _expand_graph_with_node_name(penman_graph, edge.source, node_names, seen_node_names)
    return ret + ")"


if __name__ == "__main__":
    # graph = penman.decode('(a / name :op1 "James" :op2 "Potter")')
    # print(penman.encode(graph))
    # turn_attributes_into_nodes(graph)
    # print(penman.encode(graph))
    # load_corpus_with_alignments(LEAMR_LDC_TRAIN_AMRS, LEAMR_ALIGNMENTS)
    graph = penman.decode("( l / love-01 :ARG0 (g / girl) :ARG1 (f / foonch :mod (i / ingenious)))")
    print(graph_string_from_connected_node_names(graph, ["l", "f", "g"]))
