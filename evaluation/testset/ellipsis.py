from penman import load
import csv
from evaluation.util import get_node_by_name
from evaluation.file_utils import load_corpus_from_folder


def evaluate_ellipsis(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    prereq_recalled, recalled, total = get_ellipsis_success_counts(gold_amrs, parser_name, predicted_amrs, root_dir)

    prereq_recall = prereq_recalled / total
    recall = recalled / total if total > 0 else 1.0

    return prereq_recall, recall


def get_ellipsis_success_counts(gold_amrs, parser_name, predicted_amrs, root_dir):
    if gold_amrs is None:
        gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")
    if predicted_amrs is None:
        predicted_amrs = load(f"{root_dir}/{parser_name}-output/testset.txt")
    id2labels = dict()
    with open(f"{root_dir}/corpus/ellipsis_filtered.tsv", "r") as f:
        csvreader = csv.reader(f, delimiter='\t', quotechar=None)
        for row in csvreader:
            graph_id = row[0]
            ellipsis_label = row[1]
            ellipsis_labels_here = id2labels.setdefault(graph_id, [])
            ellipsis_labels_here.append(ellipsis_label)
    total = 0
    prereq_recalled = 0
    recalled = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels:
            total += len(id2labels[gold_amr.metadata['id']])
            for ellipsis_label in id2labels[gold_amr.metadata['id']]:
                count = len([instance for instance in predicted_amr.instances() if instance.target == ellipsis_label])
                if count >= 1:
                    prereq_recalled += 1
                if count >= 2:
                    recalled += 1
    return prereq_recalled, recalled, total


def main():
    print(evaluate_ellipsis(parser_name="amparser"))


if __name__ == "__main__":
    main()
