import argparse

from penman import load
from scripts.argparse_formatter import SmartFormatter
from evaluation.full_evaluation.run_full_evaluation import display_results, \
    store_results, get_results, load_predictions, display_and_store_averages, display_and_store_by_size
from evaluation.full_evaluation.category_evaluation.category_metadata import *
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo


# TODO  seems to be a problem with the encoded tsv: has old ids


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all categories.", formatter_class=SmartFormatter)
    parser.add_argument('-gt', '--gold_amr_testset_file', type=str, help='Path to gold AMR file (testset). A single '
                                                                         'file containing all AMRs of the AMRBank 3.0'
                                                                         'testset.')
    parser.add_argument('-gg', '--gold_amr_grapes_file', type=str, help='Path to GrAPES gold corpus file. Optional, default corpus/corpus.txt', default="corpus/corpus.txt")
    parser.add_argument('-pt', '--predicted_amr_testset_file', type=str,
                        help='Path to predicted AMR 3.0 testset file. Must contain AMRs '
                             'for all sentences in the gold file, in the same '
                             'order.')
    parser.add_argument('-pg', '--predicted_amr_grapes_file', type=str,
                        help='Path to predicted AMR file. Must contain AMRs '
                             'for all sentences in the gold grapes corpus file, in the same '
                             'order.')
    parser.add_argument('--all_metrics', action='store_true', help='If set, all metrics will be computed. If not set,'
                                                                   ' only the metrics that are used in the paper will be'
                                                                   ' computed. Affected metrics are Smatch for'
                                                                   ' structural generalization, and unlabeled edge '
                                                                   'attachment scores.')
    parser.add_argument('-b', '--bunch', type=int, required=False, default=None, help='Only evaluate this "bunch" of categories. Optional.'
                                                        ' Choose a number from the following:\n'
                                                        + get_formatted_category_names([get_bunch_display_name_for_number(i) for i in range(1, len(bunch_number2name))]))
    parser.add_argument('-n', "--parser_name", type=str, required=False, default=None, help='Parser name. Optional, used for output storage. ')
    parser.add_argument('-x', "--strict", action='store_true', required=False, default=False, help='Strict mode: fail if any errors encountered')
    parser.add_argument("-e", "--error_analysis", action="store_true", help="Pickle correct and incorrect graph ids")
    parser.add_argument("-s", "--smatch", action="store_true", help="Calculate Smatch for all categories (warning: slow)")

    args = parser.parse_args()
    return args

def instance_info_from_args(args):
    instance_instructions = EvaluationInstanceInfo(
        path_to_full_testset_predictions_file_from_root=args.predicted_amr_testset_file,
        path_to_grapes_predictions_file_from_root=args.predicted_amr_grapes_file,
        path_to_gold_testset_file_from_root=args.gold_amr_testset_file,
        do_error_analysis=args.error_analysis,
        verbose_error_analysis=False,
        parser_name=args.parser_name,
        run_smatch=args.smatch,
        fail_ok=-1 if args.strict else 0,
        print_f1_default=args.all_metrics,
        print_unlabeled_edge_attachment=args.all_metrics,
        run_full_corpus_smatch=args.smatch,
        run_structural_generalisation_smatch=args.all_metrics,
    )
    return instance_instructions


def figure_out_what_to_run(gold_graphs_grapes, gold_graphs_testset, predicted_graphs_grapes, predicted_graphs_testset):
    """
    Use what we got from the prser arguments to figure out which evaluations to run
    Args:
        gold_graphs_grapes:
        gold_graphs_testset:
        predicted_graphs_grapes:
        predicted_graphs_testset:

    Returns:
        use_grapes, use_grapes_from_ptb, use_grapes_from_testset, use_testset
    """
    # initialise: will determine whether to print averages for their sets
    full_corpus_length = 1584
    minimal_corpus_length = 1471
    unbounded_dependencies_length = 66  # PTB
    word_disambiguation_length = 47  # AMR 3.0
    use_testset = gold_graphs_testset is not None and predicted_graphs_testset is not None
    use_grapes = gold_graphs_grapes is not None and predicted_graphs_grapes is not None
    use_grapes_from_testset = use_grapes and len(gold_graphs_grapes) in [
        minimal_corpus_length + word_disambiguation_length, full_corpus_length]
    use_grapes_from_ptb = use_grapes and len(gold_graphs_grapes) in [
        minimal_corpus_length + unbounded_dependencies_length, full_corpus_length]
    if not use_testset:
        print("No testset AMRs given. Skipping testset categories.")
    if not use_grapes:
        print("No GrAPES AMRs given. Skipping GrAPES categories.")
    if not use_grapes_from_testset:
        print("No AMRs for the 'Word ambiguities (handcrafted)' category were given. Will attempt with individual files, but if it fails, will skip it, as well as the"
              " 'Lexical disambiguation' compact evaluation results. You can add the graphs from the AMR testset with a"
              " script; see the documentation on the GitHub page.")
    if not use_grapes_from_ptb:
        print("No AMRs for the 'Unbounded dependencies' category were given. Will attempt with individual files, but if it fails, will skip it, as well as the"
              " 'Edge attachments' compact evaluation results. You can add the graphs from the PTB with a"
              " script; see the documentation on the GitHub page.")
    return use_grapes, use_grapes_from_ptb, use_grapes_from_testset, use_testset


def main():

    # extract info from command line
    args = parse_args()
    instance_info = instance_info_from_args(args)

    # read in the graphs from the given paths
    if instance_info.gold_testset_path() is not None and instance_info.testset_pred_file_path() is not None:
        gold_graphs_testset = load(instance_info.gold_testset_path(), encoding="utf8")
        predicted_graphs_testset = load_predictions(instance_info.testset_pred_file_path(), encoding="utf8")
        if len(gold_graphs_testset) != len(predicted_graphs_testset):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the testset here."
                "Got " + str(len(gold_graphs_testset)) + " gold AMRs and " + str(len(predicted_graphs_testset))
                + " predicted AMRs.")
    else:
        gold_graphs_testset = predicted_graphs_testset = None

    if instance_info.full_grapes_pred_file_path():
        gold_graphs_grapes = load(instance_info.gold_grapes_path(), encoding="utf8")
        predicted_graphs_grapes = load_predictions(instance_info.full_grapes_pred_file_path(), encoding="utf8")
        # predictions_directory = os.path.dirname(args.predicted_amr_grapes_file)

        if len(gold_graphs_grapes) != len(predicted_graphs_grapes):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the grapes corpus here."
                "Got " + str(len(gold_graphs_grapes)) + " gold AMRs and " + str(len(predicted_graphs_grapes))
                + " predicted AMRs.")
    else:
        gold_graphs_grapes = predicted_graphs_grapes = None

    use_grapes, use_grapes_from_ptb, use_grapes_from_testset, use_testset = figure_out_what_to_run(gold_graphs_grapes,
                                                                                                   gold_graphs_testset,
                                                                                                   predicted_graphs_grapes,
                                                                                                   predicted_graphs_testset)
    if instance_info.run_smatch:
        print("We will run Smatch on all categories. This may take a while...\n"
              " to avoid this, stop and rerun without the --smatch option.")
    # run the evaluation
    results, by_size, sums, divisors, dont_print_these_averages = get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset,
                                   predicted_graphs_grapes, instance_info, use_grapes, use_grapes_from_ptb,
                                   use_grapes_from_testset, use_testset, bunch=args.bunch)

    results_dir = f"{instance_info.root_dir}/data/processed/results/from_evaluate_all_categories"

    store_results(results, instance_info, results_dir=results_dir)
    display_results(results, bunch=args.bunch)
    display_and_store_averages(divisors, instance_info.parser_name, results_dir, sums, dont_print_these_averages, args.bunch)
    display_and_store_by_size(by_size, instance_info.parser_name, results_dir)
    # table = structural_generalisation_by_size_as_table(by_size)
    # out_csv_by_size = f"{results_dir}/{instance_info.parser_name}_by_size.csv"
    # csv.writer(open(out_csv_by_size, "w")).writerow(table.field_names)
    # csv.writer(open(out_csv_by_size, "a", encoding="utf8")).writerows(table.rows)


    if instance_info.do_error_analysis:
        print("Error analysis pickles in", f"{instance_info.root_dir}/error_analysis/{instance_info.parser_name}/")


if __name__ == "__main__":
    main()

    # for key in category_name_to_set_class_and_metadata:
    #     if category_name_to_set_class_and_metadata[key][1].display_name != category_name_to_print_name[key]:
    #         print(key, category_name_to_set_class_and_metadata[key][1].display_name, category_name_to_print_name[key])
