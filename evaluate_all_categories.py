import argparse
import csv
import os

from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.run_full_evaluation import get_arguments_for_evaluation_class
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.single_eval import num_to_score

from evaluation.category_metadata import category_name_to_set_class_and_metadata, category_name_to_print_name
from prettytable import PrettyTable


# TODO wrong:
# Other seen entities                    | Recall                | 78    | 72          | 83          | 237
# should be 80
# | 5   | Other unseen entities                  | Recall                | 70    | 61          | 78          | 109         |
# should be 74
# | 7   | Word ambiguities (Karidi et al., 2021) | Recall                | 83    | 81          | 84          | 1471        |
# Should have 95 total! Also should be 75
# Also: seems to be a problem with the encoded tsv: has old ids



set_names_with_category_names = [
    ("1. Pragmatic reentrancies", ["pragmatic_coreference_testset", "pragmatic_coreference_winograd"]),
    ("2. Unambiguous reentrancies", ["syntactic_gap_reentrancies", "unambiguous_coreference"]),
    ("3. Structural generalization",
     ["nested_control_and_coordination", "nested_control_and_coordination_sanity_check",
      "multiple_adjectives", "multiple_adjectives_sanity_check",
      "centre_embedding", "centre_embedding_sanity_check",
      "cp_recursion", "cp_recursion_sanity_check",
      "cp_recursion_plus_coreference", "cp_recursion_plus_coreference_sanity_check",
      "cp_recursion_plus_rc", "cp_recursion_plus_rc_sanity_check",
      "cp_recursion_plus_rc_plus_coreference", "cp_recursion_plus_rc_plus_coreference_sanity_check",
      "long_lists", "long_lists_sanity_check"
      ]),
    ("4. Rare and unseen words",
     ["rare_node_labels", "unseen_node_labels", "rare_predicate_senses_excl_01", "unseen_predicate_senses_excl_01",
      "rare_edge_labels_ARG2plus", "unseen_edge_labels_ARG2plus"]),
    ("5. Special entities",
     ["seen_names", "unseen_names", "seen_dates", "unseen_dates", "other_seen_entities", "other_unseen_entities"]),
    ("6. Entity classification and linking",
     ["types_of_seen_named_entities", "types_of_unseen_named_entities", "seen_andor_easy_wiki_links",
      "hard_unseen_wiki_links"]),
    ("7. Lexical disambiguations",
     ["frequent_predicate_senses_incl_01", "word_ambiguities_handcrafted", "word_ambiguities_karidi_et_al_2021"]),
    ("8. Edge attachments", ["pp_attachment", "unbounded_dependencies", "passives", "unaccusatives"]),
    ("9. Non-trivial word-to-node relations", ["ellipsis", "multinode_word_meanings", "imperatives"])
    ]

category_names_to_source_corpus_name = {
    "pragmatic_coreference_testset": "testset",
    "pragmatic_coreference_winograd": "grapes",
    "syntactic_gap_reentrancies": "testset",
    "unambiguous_coreference": "testset",
    "nested_control_and_coordination": "grapes",
    "nested_control_and_coordination_sanity_check": "grapes",
    "multiple_adjectives": "grapes",
    "multiple_adjectives_sanity_check": "grapes",
    "centre_embedding": "grapes",
    "centre_embedding_sanity_check": "grapes",
    "cp_recursion": "grapes",
    "cp_recursion_sanity_check": "grapes",
    "cp_recursion_plus_coreference": "grapes",
    "cp_recursion_plus_coreference_sanity_check": "grapes",
    "cp_recursion_plus_rc": "grapes",
    "cp_recursion_plus_rc_sanity_check": "grapes",
    "cp_recursion_plus_rc_plus_coreference": "grapes",
    "cp_recursion_plus_rc_plus_coreference_sanity_check": "grapes",
    "long_lists": "grapes",
    "long_lists_sanity_check": "grapes",
    "rare_node_labels": "testset",
    "unseen_node_labels": "testset",
    "rare_predicate_senses_excl_01": "testset",
    "unseen_predicate_senses_excl_01": "grapes",
    "rare_edge_labels_ARG2plus": "testset",
    "unseen_edge_labels_ARG2plus": "grapes",
    "seen_names": "testset",
    "unseen_names": "testset",
    "seen_dates": "testset",
    "unseen_dates": "testset",
    "other_seen_entities": "testset",
    "other_unseen_entities": "testset",
    "types_of_seen_named_entities": "testset",
    "types_of_unseen_named_entities": "testset",
    "seen_andor_easy_wiki_links": "testset",
    "hard_unseen_wiki_links": "testset",
    "frequent_predicate_senses_incl_01": "testset",
    "word_ambiguities_handcrafted": "grapes_from_testset",
    "word_ambiguities_karidi_et_al_2021": "grapes",
    "pp_attachment": "grapes",
    "unbounded_dependencies": "grapes_from_ptb",
    "passives": "testset",
    "unaccusatives": "testset",
    "ellipsis": "testset",
    "multinode_word_meanings": "testset",
    "imperatives": "testset"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all categories.")
    parser.add_argument('-gt', '--gold_amr_testset_file', type=str, help='Path to gold AMR file (testset). A single '
                                                                         'file containing all AMRs of the AMRBank 3.0'
                                                                         'testset.')
    parser.add_argument('-gg', '--gold_amr_grapes_file', type=str, help='Path to GrAPES gold corpus file')
    parser.add_argument('-pt', '--predicted_amr_testset_file', type=str,
                        help='Path to predicted AMR file. Must contain AMRs '
                             'for all sentences in the gold file, in the same '
                             'order.')
    parser.add_argument('-pg', '--predicted_amr_grapes_file', type=str,
                        help='Path to predicted AMR file. Must contain AMRs '
                             'for all sentences in the gold grapes corpus file, in the same '
                             'order.')
    parser.add_argument('--all_metrics', action='store_true', help='If set, all metrics will be computed. If not set,'
                                                                   'only the metrics that are used in the paper will be'
                                                                   'computed. Affected metrics are Smatch for'
                                                                   ' structural generalization, and unlabeled edge '
                                                                   'attachment scores.')
    args = parser.parse_args()
    return args


def get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes, predictions_directory,
                filter_out_f1=True, filter_out_unlabeled_edge_attachment=True):
    """
    Returns a list of result rows. Each row has the following format:
    [set number, category name, metric name, score, lower_bound, upper_bound, sample_size]
    (the latter three are omitted for f-score results, since they don't apply there)
    """
    full_corpus_length = 1584
    minimal_corpus_length = 1471
    unbounded_dependencies_length = 66  # PTB
    word_disambiguation_length = 47  # AMR 3.0
    use_testset = gold_graphs_testset is not None and predicted_graphs_testset is not None
    use_grapes = gold_graphs_grapes is not None and predicted_graphs_grapes is not None
    use_grapes_from_testset = use_grapes and len(gold_graphs_grapes) in [minimal_corpus_length + word_disambiguation_length, full_corpus_length]
    use_grapes_from_ptb = use_grapes and len(gold_graphs_grapes) in [minimal_corpus_length + unbounded_dependencies_length, full_corpus_length]
    if not use_testset:
        print("No testset AMRs given. Skipping testset categories.")
    if not use_grapes:
        print("No GrAPES AMRs given. Skipping GrAPES categories.")
    if not use_grapes_from_testset:
        print("No AMRs for the 'Word ambiguities (handcrafted)' category were given. Will skip it, as well as the"
              " 'Lexical disambiguation' compact evaluation results. You can add the graphs from the AMR testset with a"
              " script; see the documentation on the GitHub page.")
    if not use_grapes_from_ptb:
        print("No AMRs for the 'Unbounded dependencies' category were given. Will skip it, as well as the"
              " 'Edge attachments' compact evaluation results. You can add the graphs from the PTB with a"
              " script; see the documentation on the GitHub page.")

    results = []
    struct_gen_by_size = {}
    for set_name, category_names in set_names_with_category_names:
        print("Evaluating " + set_name)
        for category_name in category_names:
            if do_skip_category(category_name, use_testset, use_grapes, use_grapes_from_testset, use_grapes_from_ptb):
                try:
                    # try to get the subcorpus from the same folder as the full corpus
                    eval_class, info = category_name_to_set_class_and_metadata[category_name]
                    eval_args = get_arguments_for_evaluation_class(info, predictions_directory, "parser", ".")
                    print("\n ### Trying skipped category with args", eval_args[-1])
                    print(f"gold len {len(eval_args[0])}, predicted len {len(eval_args[1])}")
                    set = eval_class(*eval_args)
                    results_here = set.run_evaluation()
                    rows = make_rows_for_results(category_name, filter_out_f1, filter_out_unlabeled_edge_attachment,
                                                 results_here, set_name)
                    results.extend(rows)
                except Exception as e:
                    print(f"Can't get category {category_name}, error: {e}")
                    # raise e
                    results.append(make_empty_result(set_name, category_name_to_print_name[category_name]))
            else:
                set_class, info = category_name_to_set_class_and_metadata[category_name]
                if info.subcorpus_filename is None:  # testset
                    gold_graphs = gold_graphs_testset
                    predicted_graphs = predicted_graphs_testset
                else:
                    gold_graphs = gold_graphs_grapes
                    predicted_graphs = predicted_graphs_grapes

                evaluator = set_class(gold_graphs, predicted_graphs, "parser", ".", info)
                results_here = evaluator.run_evaluation()
                rows = make_rows_for_results(category_name, filter_out_f1, filter_out_unlabeled_edge_attachment,
                                      results_here, set_name)
                results.extend(rows)
                if info.subtype == "structural_generalization":
                    by_size = evaluator.get_results_by_size()
                    struct_gen_by_size[info.display_name] = by_size

    return results, struct_gen_by_size


def make_rows_for_results(category_name, filter_out_f1, filter_out_unlabeled_edge_attachment, results_here,
                          set_name):
    print("Include smatch?", not filter_out_f1)
    rows = []
    for r in results_here:
        metric_name = r[1]
        if filter_out_f1 and metric_name == "Smatch":
            continue
        if filter_out_unlabeled_edge_attachment and metric_name == "Unlabeled edge recall":
            continue
        metric_type = r[2]
        print("metric_type", metric_type)
        if metric_type == EVAL_TYPE_SUCCESS_RATE:
            wilson_ci = wilson_score_interval(r[3], r[4])
            if r[4] > 0:
                rows.append([set_name[0], category_name_to_print_name[category_name], metric_name,
                                num_to_score(r[3] / r[4]),
                                num_to_score(wilson_ci[0]),
                                num_to_score(wilson_ci[1]),
                                r[4]])
            else:
                print(
                    "ERROR: Division by zero! This means something unexpected went wrong (feel free to contact the "
                    "developers of GrAPES for help, e.g. by filing an issue on GitHub).")
                print(r)
        elif metric_type == EVAL_TYPE_F1:
            print("Found Smatch!")
            rows.append([set_name[0], category_name_to_print_name[category_name], metric_name,
                            num_to_score(r[3]), "N/A", "N/A", "N/A"])
        else:
            print(
                "ERROR: Unexpected evaluation type! This means something unexpected went wrong (feel free to "
                "contact the developers of GrAPES for help, e.g. by filing an issue on GitHub).")
            print(r)
    return rows


def make_empty_result(set_name, category_name):
    return [set_name[0], category_name, "N/A", "N/A", "N/A", "N/A", "N/A"]


def do_skip_category(category_name, use_testset, use_grapes, use_grapes_from_testset, use_grapes_from_ptb):
    if not use_testset and category_names_to_source_corpus_name[category_name] == "testset":
        return True
    if not use_grapes and category_names_to_source_corpus_name[category_name] == "grapes":
        return True
    if not use_grapes_from_testset and category_names_to_source_corpus_name[category_name] == "grapes_from_testset":
        return True
    if not use_grapes_from_ptb and category_names_to_source_corpus_name[category_name] == "grapes_from_ptb":
        return True
    return False


def main():
    parser = "parser"

    args = parse_args()
    if args.gold_amr_testset_file is not None and args.predicted_amr_testset_file is not None:
        gold_graphs_testset = load(args.gold_amr_testset_file, encoding="utf8")
        predicted_graphs_testset = load(args.predicted_amr_testset_file)
        if len(gold_graphs_testset) != len(predicted_graphs_testset):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the testset here."
                "Got " + str(len(gold_graphs_testset)) + " gold AMRs and " + str(len(predicted_graphs_testset))
                + " predicted AMRs.")
    else:
        gold_graphs_testset = predicted_graphs_testset = None

    if args.gold_amr_grapes_file is not None and args.predicted_amr_grapes_file is not None:
        gold_graphs_grapes = load(args.gold_amr_grapes_file, encoding="utf8")
        predicted_graphs_grapes = load(args.predicted_amr_grapes_file, encoding="utf8")
        predictions_directory = os.path.dirname(args.predicted_amr_grapes_file)

        if len(gold_graphs_grapes) != len(predicted_graphs_grapes):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the grapes corpus here."
                "Got " + str(len(gold_graphs_grapes)) + " gold AMRs and " + str(len(predicted_graphs_grapes))
                + " predicted AMRs.")

    else:
        gold_graphs_grapes = predicted_graphs_grapes = predictions_directory = None

    results, by_size = get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes,
                          predictions_directory,
                          filter_out_f1=not args.all_metrics, filter_out_unlabeled_edge_attachment=not args.all_metrics)
    out_dir = f"data/processed/results"
    os.makedirs(out_dir, exist_ok=True)
    csv.writer(open(f"{out_dir}/results.csv", "w", encoding="utf8")).writerows(results)

    print_table = PrettyTable(field_names=["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"])
    print_table.align = "l"
    for row in results:
        print_table.add_row(row)

    pretty_print_structural_generalisation_by_size(by_size)

    print("\nAll results")
    print(print_table)


def pretty_print_structural_generalisation_by_size(results):
    """
    Prints the structural generalisation results split up by size
    Args:
        results: dict from parser name to dataset name to dict from size to score
    """
    print(results)
    from prettytable import PrettyTable
    table = PrettyTable()
    max_size = 10
    field_names = ["Dataset"]
    for n in range(1, max_size + 1):
        field_names.append(str(n))
    table.field_names = field_names
    table.align = "l"
    for dataset in results:
        sizes = results[dataset].keys()
        row = [dataset]
        for n in range(1, max_size + 1):
            if n in sizes:
                row.append(results[dataset][n])
            else:
                row.append("")
        table.add_row(row)

    print("\nStructure generalisation results by size")
    print(table)


if __name__ == "__main__":
    main()

    # for key in category_name_to_set_class_and_metadata:
    #     if category_name_to_set_class_and_metadata[key][1].display_name != category_name_to_print_name[key]:
    #         print(key, category_name_to_set_class_and_metadata[key][1].display_name, category_name_to_print_name[key])
