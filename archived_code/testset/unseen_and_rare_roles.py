from penman import load
import csv
from evaluation.util import get_node_by_name
from evaluation.file_utils import load_corpus_from_folder


def evaluate_rare_roles(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return evaluate_roles(gold_amrs, predicted_amrs, parser_name, root_dir, tsv_file_name="rare_roles.tsv")


def evaluate_unseen_roles(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return evaluate_roles(gold_amrs, predicted_amrs, parser_name, root_dir, tsv_file_name="unseen_roles.tsv")


def evaluate_roles(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../", tsv_file_name=None):
    if gold_amrs is None:
        gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")
    if predicted_amrs is None:
        predicted_amrs = load(f"{root_dir}/{parser_name}-output/testset.txt")
    id2labels = dict()
    with open(f"{root_dir}/corpus/{tsv_file_name}", "r") as f:
        csvreader = csv.reader(f, delimiter='\t', quotechar=None)
        for row in csvreader:
            graph_id = row[0]
            edge_triple = row[1:4]
            edge_triples_here = id2labels.setdefault(graph_id, [])
            edge_triples_here.append(edge_triple)

    nodes_recalled = 0
    unlabeled_edges_recalled = 0
    edges_recalled = 0
    total = 0
    nodes_recalled_arg2plus = 0
    unlabeled_edges_recalled_arg2plus = 0
    edges_recalled_arg2plus = 0
    total_arg2plus = 0

    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels:
            total += len(id2labels[gold_amr.metadata['id']])
            for edge_triple in id2labels[gold_amr.metadata['id']]:
                # print(edge_triple)
                if not edge_triple[1] == ":ARG0" and not edge_triple[1] == ":ARG1":
                    total_arg2plus += 1
                all_node_labels = [instance.target for instance in predicted_amr.instances()]
                if edge_triple[0] in all_node_labels and edge_triple[2] in all_node_labels:
                    nodes_recalled += 1
                    if not edge_triple[1] == ":ARG0" and not edge_triple[1] == ":ARG1":
                        nodes_recalled_arg2plus += 1
                for edge in predicted_amr.edges():
                    if edge_triple[0] == get_node_by_name(edge.source, predicted_amr).target\
                            and edge_triple[2] == get_node_by_name(edge.target, predicted_amr).target:
                        unlabeled_edges_recalled += 1
                        if not edge_triple[1] == ":ARG0" and not edge_triple[1] == ":ARG1":
                            unlabeled_edges_recalled_arg2plus += 1
                        break
                for edge in predicted_amr.edges(role=edge_triple[1]):
                    if edge_triple[0] == get_node_by_name(edge.source, predicted_amr).target\
                            and edge_triple[2] == get_node_by_name(edge.target, predicted_amr).target:
                        edges_recalled += 1
                        if not edge_triple[1] == ":ARG0" and not edge_triple[1] == ":ARG1":
                            edges_recalled_arg2plus += 1
                        break

    node_recall = nodes_recalled / total if total > 0 else 1.0
    unlabeled_edge_recall = unlabeled_edges_recalled / total if total > 0 else 1.0
    edge_recall = edges_recalled / total if total > 0 else 1.0
    node_recall_arg2plus = nodes_recalled_arg2plus / total_arg2plus if total_arg2plus > 0 else 1.0
    unlabeled_edge_recall_arg2plus = unlabeled_edges_recalled_arg2plus / total_arg2plus if total_arg2plus > 0 else 1.0
    edge_recall_arg2plus = edges_recalled_arg2plus / total_arg2plus if total_arg2plus > 0 else 1.0

    return node_recall, unlabeled_edge_recall, edge_recall, node_recall_arg2plus, unlabeled_edge_recall_arg2plus, \
           edge_recall_arg2plus


def main():
    print(evaluate_rare_roles(parser_name="amrbart"))
    print(evaluate_unseen_roles(parser_name="amrbart"))


if __name__ == "__main__":
    main()
