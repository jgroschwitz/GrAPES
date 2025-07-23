from evaluation.graph_matcher import check_fragment_existence, check_edge_label_existence
from evaluation.corpus_metrics import run_checks_and_get_backup_data_if_applicable, check_edge_existence
from evaluation.file_utils import read_tsv_with_comments

EDGE_TYPE = "EDGE"
EDGE_PLUS_TYPE = "EDGE_PLUS"
NODE_TYPE = "NODE"


def evaluate_word_ambiguities_test(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    gold_amrs, predicted_amrs = run_checks_and_get_backup_data_if_applicable(False, gold_amrs, parser_name,
                                                                             predicted_amrs, root_dir, False,
                                                                             False)
    with open(root_dir + "corpus/word_ambiguities_from_test.tsv") as f:
        csv_reader = read_tsv_with_comments(f)
        id2rows = dict( )
        for row in csv_reader:
            labels_here = id2rows.setdefault(row[0], [])
            labels_here.append(row)

        prerequisite_count = 0
        success_count = 0
        total_count = 0

        for gold_graph, predicted_graph in zip(gold_amrs, predicted_amrs):
            if gold_graph.metadata['id'] in id2rows:
                for row in id2rows[gold_graph.metadata['id']]:
                    # graph_id = row[0]
                    # word = row[1]
                    gold_type = row[2]
                    if gold_type == EDGE_TYPE:
                        source = row[3]
                        target = row[5]
                        edge_label = row[4]
                        prerequisite = check_fragment_existence(source, predicted_graph)\
                                       and check_fragment_existence(target, predicted_graph)
                        success = check_edge_existence([source, edge_label, target], predicted_graph)
                    elif gold_type == EDGE_PLUS_TYPE:
                        source = row[3]
                        target = row[5]
                        edge_label = row[4]
                        # for EDGE_PLUS, target is part of the actual evaluation, not the prerequisite
                        prerequisite = check_fragment_existence(source, predicted_graph)
                        success = check_edge_existence([source, edge_label, target], predicted_graph)
                    elif gold_type == NODE_TYPE:
                        prerequisite = True  # no prerequisites to check for nodes
                        node_label = row[3]
                        success = check_fragment_existence(node_label, predicted_graph)
                        if not success and len(row) > 4:
                            edge_label = row[4]
                            success = check_edge_label_existence(edge_label, predicted_graph)
                    else:
                        raise ValueError("Unknown type: " + gold_type)

                    if prerequisite:
                        prerequisite_count += 1
                        if success:
                            success_count += 1
                    total_count += 1

    if total_count == 0:
        return 1.0, 1.0
    else:
        return prerequisite_count / total_count, success_count / total_count
