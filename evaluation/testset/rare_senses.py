from evaluation.corpus_metrics import calculate_node_label_recall
from evaluation.file_utils import load_corpus_from_folder
from penman import load
import csv


def evaluate_rare_senses_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    prereq = calculate_node_label_recall("rare_senses_filtered.tsv", gold_amrs, predicted_amrs, parser_name, root_dir,
                                         use_sense=False)
    recall = calculate_node_label_recall("rare_senses_filtered.tsv", gold_amrs, predicted_amrs, parser_name, root_dir,
                                         use_sense=True)
    return prereq, recall


def main():
    print(evaluate_rare_senses_test(parser_name="amrbart"))
    print(evaluate_rare_senses_test(parser_name="amparser"))


if __name__ == '__main__':
    main()
