from evaluation.corpus_metrics import calculate_edge_recall_for_tsv_file


def evaluate_pragmatic_reentrancies_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return calculate_edge_recall_for_tsv_file("reentrancies_pragmatic_filtered.tsv", gold_amrs, predicted_amrs, parser_name, root_dir,
                                              use_sense=False, parent_column=4, parent_edge_column=5)


def evaluate_syntactic_gap_reentrancies_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return calculate_edge_recall_for_tsv_file("reentrancies_syntactic_gap_filtered.tsv", gold_amrs, predicted_amrs, parser_name, root_dir,
                                              use_sense=False, parent_column=4, parent_edge_column=5)


def evaluate_unambiguous_coreference_reentrancies_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    return calculate_edge_recall_for_tsv_file("reentrancies_unambiguous_coreference_filtered.tsv", gold_amrs, predicted_amrs, parser_name, root_dir,
                                              use_sense=False, parent_column=4, parent_edge_column=5)


def main():
    print("pragmatic (AM parser then AMRBART)")
    print(evaluate_pragmatic_reentrancies_test(parser_name="amparser"))
    print(evaluate_pragmatic_reentrancies_test(parser_name="amrbart"))
    print()
    print("syntactic gap (AM parser then AMRBART)")
    print(evaluate_syntactic_gap_reentrancies_test(parser_name="amparser"))
    print(evaluate_syntactic_gap_reentrancies_test(parser_name="amrbart"))
    print()
    print("unambiguous coreference (AM parser then AMRBART)")
    print(evaluate_unambiguous_coreference_reentrancies_test(parser_name="amparser"))
    print(evaluate_unambiguous_coreference_reentrancies_test(parser_name="amrbart"))


if __name__ == '__main__':
    main()
