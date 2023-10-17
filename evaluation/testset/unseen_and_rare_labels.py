from evaluation.util import strip_sense
from evaluation.corpus_metrics import calculate_node_label_recall
from evaluation.file_utils import load_corpus_from_folder
from penman import load
import csv


def evaluate_unseen_labels_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return calculate_node_label_recall("unseen_node_labels_test_filtered.tsv", gold_amrs, predicted_amrs, parser_name, root_dir,
                                       use_sense=True, error_analysis_output_filename="unseen_node_labels.pkl",
                                       error_analysis_message="Focus only on the content represented by the highlighted node!")


def evaluate_rare_labels_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return calculate_node_label_recall("rare_node_labels_test.tsv", gold_amrs, predicted_amrs, parser_name, root_dir,
                                       use_sense=True)


def main():
    print(evaluate_unseen_labels_test(parser_name="amrbart"))
    print(evaluate_rare_labels_test(parser_name="amrbart"))


if __name__ == '__main__':
    main()
