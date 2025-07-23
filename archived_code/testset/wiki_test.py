from evaluation.corpus_metrics import calculate_node_label_recall
from penman import load
import csv


def evaluate_hard_wiki_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return calculate_node_label_recall("hard_wiki_test_data.tsv", gold_amrs, predicted_amrs, parser_name,
                                       root_dir, use_attributes=True, attribute_label=":wiki")


def evaluate_seen_andor_easy_wiki_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return calculate_node_label_recall("seen_andor_easy_wiki_test_data.tsv", gold_amrs, predicted_amrs, parser_name,
                                       root_dir, use_attributes=True, attribute_label=":wiki")


def main():
    print(evaluate_hard_wiki_test(parser_name="amrbart"))


if __name__ == '__main__':
    main()
