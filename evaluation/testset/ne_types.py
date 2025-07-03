from evaluation.corpus_metrics import calculate_node_label_recall, graph_is_in_ids
from evaluation.file_utils import load_corpus_from_folder
from evaluation.util import get_name, get_node_by_name
from penman import load
import csv


def evaluate_ne_types_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    assert predicted_amrs is not None or parser_name is not None
    if gold_amrs is None:
        gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")
    if predicted_amrs is None:
        predicted_amrs = load(f"{root_dir}/{parser_name}-output/testset.txt")

    unseen_entity_recall, unseen_name_recall = \
        get_ne_type_recall_and_prereq(f"{root_dir}/corpus/unseen_ne_types_test.tsv", gold_amrs, predicted_amrs)

    seen_entity_recall, seen_name_recall = \
        get_ne_type_recall_and_prereq(f"{root_dir}/corpus/seen_ne_types_test.tsv", gold_amrs, predicted_amrs)

    return unseen_name_recall, unseen_entity_recall, seen_name_recall, seen_entity_recall


def get_ne_type_recall_and_prereq(filename, gold_amrs, predicted_amrs):
    name_recalled, type_recalled, total = get_ne_type_successes_and_sample_size(filename, gold_amrs, predicted_amrs)
    name_recall = name_recalled / total if total > 0 else 1.0
    entity_recall = type_recalled / total if total > 0 else 1.0
    return entity_recall, name_recall


def get_ne_type_successes_and_sample_size(id2labels, gold_amrs, predicted_amrs):
    """
    Expects to find graph ID in column 0, type in column 1, and name in column 2
    :param filename:
    :param gold_amrs:
    :param predicted_amrs:
    :return: Total prereq counts (named entity exists), total success counts (NE has correct type) and total sample size.
    """
    name_recalled = 0
    type_recalled = 0
    total = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if graph_is_in_ids(gold_amr, id2labels.keys()):
            total += len(id2labels[gold_amr.metadata['id']])
            for type_and_name in id2labels[gold_amr.metadata['id']]:
                for edge in predicted_amr.edges(role=":name"):
                    entity_label = get_node_by_name(edge.source, predicted_amr).target
                    name_string = get_name(edge.target, predicted_amr)
                    if name_string == type_and_name[1]:
                        name_recalled += 1
                        # print(entity_label, type_and_name[0])
                        if entity_label == type_and_name[0]:
                            type_recalled += 1
                        break
    return name_recalled, type_recalled, total


def get_2_columns_from_tsv_by_id(filename, id_column=0, column_1=1, column_2=2):
    id2labels = dict()
    with open(filename, "r") as f:
        csvreader = csv.reader(f, delimiter='\t', quotechar=None)
        for row in csvreader:
            graph_id = row[id_column]
            ne_type = row[column_1]
            name_string = row[column_2]
            labels_here = id2labels.setdefault(graph_id, [])
            labels_here.append((ne_type, name_string))
    return id2labels


def main():
    print(evaluate_ne_types_test(parser_name="amrbart"))


if __name__ == '__main__':
    main()
