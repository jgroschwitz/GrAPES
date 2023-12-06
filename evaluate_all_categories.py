import argparse
import csv

from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.i_pragmatic_reentrancies import PragmaticReentrancies
from evaluation.full_evaluation.category_evaluation.ii_unambiguous_reentrancies import UnambiguousReentrancies
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.single_eval import num_to_score

from evaluate_single_category import category_name_to_set_class_and_eval_function
from prettytable import PrettyTable

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

category_name_to_print_name = {
    "pragmatic_coreference_testset": "Pragmatic coreference (testset)",
    "pragmatic_coreference_winograd": "Pragmatic coreference (Winograd)",
    "syntactic_gap_reentrancies": "Syntactic (gap) reentrancies",
    "unambiguous_coreference": "Unambiguous coreference",
    "nested_control_and_coordination": "Nested control and coordination",
    "nested_control_and_coordination_sanity_check": "Sanity check",
    "multiple_adjectives": "Multiple adjectives",
    "multiple_adjectives_sanity_check": "Sanity check",
    "centre_embedding": "Centre embedding",
    "centre_embedding_sanity_check": "Sanity check",
    "cp_recursion": "CP recursion",
    "cp_recursion_sanity_check": "Sanity check",
    "cp_recursion_plus_coreference": "CP recursion + coreference",
    "cp_recursion_plus_coreference_sanity_check": "Sanity check",
    "cp_recursion_plus_rc": "CP recursion + relative clause (RC)",
    "cp_recursion_plus_rc_sanity_check": "Sanity check",
    "cp_recursion_plus_rc_plus_coreference": "CP recursion + RC + coreference",
    "cp_recursion_plus_rc_plus_coreference_sanity_check": "Sanity check",
    "long_lists": "Long lists",
    "long_lists_sanity_check": "Sanity check",
    "rare_node_labels": "Rare node labels",
    "unseen_node_labels": "Unseen node labels",
    "rare_predicate_senses_excl_01": "Rare predicate senses (excl. -01)",
    "unseen_predicate_senses_excl_01": "Unseen predicate senses (excl. -01)",
    "rare_edge_labels_ARG2plus": "Rare edge labels (ARG2+)",
    "unseen_edge_labels_ARG2plus": "Unseen edge labels (ARG2+)",
    "seen_names": "Seen names",
    "unseen_names": "Unseen names",
    "seen_dates": "Seen dates",
    "unseen_dates": "Unseen dates",
    "other_seen_entities": "Other seen entities",
    "other_unseen_entities": "Other unseen entities",
    "types_of_seen_named_entities": "Types of seen named entities",
    "types_of_unseen_named_entities": "Types of unseen named entities",
    "seen_andor_easy_wiki_links": "Seen and/or easy wiki links",
    "hard_unseen_wiki_links": "Hard unseen wiki links",
    "frequent_predicate_senses_incl_01": "Frequent predicate senses (incl. -01)",
    "word_ambiguities_handcrafted": "Word ambiguities (handcrafted)",
    "word_ambiguities_karidi_et_al_2021": "Word ambiguities (Karidi et al., 2021)",
    "pp_attachment": "PP attachment",
    "unbounded_dependencies": "Unbounded dependencies",
    "passives": "Passives",
    "unaccusatives": "Unaccusatives",
    "ellipsis": "Ellipsis",
    "multinode_word_meanings": "Multinode word meanings",
    "imperatives": "Imperatives"
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


def get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes,
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
    for set_name, category_names in set_names_with_category_names:
        for category_name in category_names:
            if do_skip_category(category_name, use_testset, use_grapes, use_grapes_from_testset, use_grapes_from_ptb):
                results.append(make_empty_result(set_name, category_name_to_print_name[category_name]))
            else:
                set_class, eval_function = category_name_to_set_class_and_eval_function[category_name]
                set = set_class(None, None, None, "./")
                if category_names_to_source_corpus_name[category_name] == "testset":
                    gold_graphs = gold_graphs_testset
                    predicted_graphs = predicted_graphs_testset
                else:
                    gold_graphs = gold_graphs_grapes
                    predicted_graphs = predicted_graphs_grapes
                results_here = eval_function(set, gold_graphs, predicted_graphs)
                for r in results_here:
                    metric_name = r[1]
                    if filter_out_f1 and metric_name == "Smatch":
                        continue
                    if filter_out_unlabeled_edge_attachment and metric_name == "Unlabeled edge recall":
                        continue
                    metric_type = r[2]
                    if metric_type == EVAL_TYPE_SUCCESS_RATE:
                        wilson_ci = wilson_score_interval(r[3], r[4])
                        if r[4] > 0:
                            results.append([set_name[0], category_name_to_print_name[category_name], metric_name,
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
                        results.append([set_name[0], category_name_to_print_name[category_name], metric_name,
                                        num_to_score(r[3]), "N/A", "N/A", "N/A"])
                    else:
                        print(
                            "ERROR: Unexpected evaluation type! This means something unexpected went wrong (feel free to "
                            "contact the developers of GrAPES for help, e.g. by filing an issue on GitHub).")
                        print(r)
    return results


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
    args = parse_args()
    if args.gold_amr_testset_file is not None and args.predicted_amr_testset_file is not None:
        gold_graphs_testset = load(args.gold_amr_testset_file)
        predicted_graphs_testset = load(args.predicted_amr_testset_file)
        if len(gold_graphs_testset) != len(predicted_graphs_testset):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the testset here."
                "Got " + str(len(gold_graphs_testset)) + " gold AMRs and " + str(len(predicted_graphs_testset))
                + " predicted AMRs.")
    else:
        gold_graphs_testset = predicted_graphs_testset = None

    if args.gold_amr_grapes_file is not None and args.predicted_amr_grapes_file is not None:
        gold_graphs_grapes = load(args.gold_amr_grapes_file)
        predicted_graphs_grapes = load(args.predicted_amr_grapes_file)

        if len(gold_graphs_grapes) != len(predicted_graphs_grapes):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the grapes corpus here."
                "Got " + str(len(gold_graphs_grapes)) + " gold AMRs and " + str(len(predicted_graphs_grapes))
                + " predicted AMRs.")

    else:
        gold_graphs_grapes = predicted_graphs_grapes = None

    results = get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes,
                          filter_out_f1=not args.all_metrics, filter_out_unlabeled_edge_attachment=not args.all_metrics)
    csv.writer(open("results.csv", "w")).writerows(results)

    print_table = PrettyTable(field_names=["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"])
    print_table.align = "l"
    for row in results:
        print_table.add_row(row)
    print(print_table)


if __name__ == "__main__":
    main()
