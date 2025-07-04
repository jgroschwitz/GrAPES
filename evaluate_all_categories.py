import argparse
import csv
import os
import pickle

from penman import load

from evaluate_single_category import SmartFormatter, load_predictions
from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.subcategory_info import is_grapes_category_with_testset_data, \
    is_grapes_category_with_ptb_data
from evaluation.full_evaluation.run_full_evaluation import run_single_file, evaluate, \
    pretty_print_structural_generalisation_by_size, get_root_results_path
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.util import num_to_score

from evaluation.full_evaluation.category_evaluation.category_metadata import *
from prettytable import PrettyTable

from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo


# TODO  seems to be a problem with the encoded tsv: has old ids


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all categories.", formatter_class=SmartFormatter)
    parser.add_argument('-gt', '--gold_amr_testset_file', type=str, help='Path to gold AMR file (testset). A single '
                                                                         'file containing all AMRs of the AMRBank 3.0'
                                                                         'testset.')
    parser.add_argument('-gg', '--gold_amr_grapes_file', type=str, help='Path to GrAPES gold corpus file. Optional, default corpus/corpus.txt', default="corpus/corpus.txt")
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
    parser.add_argument('-b', '--bunch', type=int, required=False, default=None, help='Only evaluate this "bunch" of categories. Optional.'
                                                        ' Choose a number from the following:\n'
                                                        + get_formatted_category_names([b for b, _ in
                                                                                        set_names_with_category_names]))
    parser.add_argument('-n', "--parser_name", type=str, required=False, default=None, help='Parser name. Optional, used for output storage. ')
    parser.add_argument('-x', "--strict", action='store_true', required=False, default=False, help='Strict mode: fail if any errors encountered')
    parser.add_argument("-e", "--error_analysis", action="store_true", help="Pickle correct and incorrect graph ids")
    parser.add_argument("-s", "--smatch", action="store_true", help="Calculate Smatch for all categories (warning: slow)")

    args = parser.parse_args()
    return args


def do_this_category(bunch, category_name):
    return bunch is None or category_name.startswith(str(bunch)+".")


def get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes,
                predictions_directory, cmd_args,
                filter_out_f1=True, filter_out_unlabeled_edge_attachment=True, bunch=None):
    """
    Returns a list of result rows. Each row has the following format:
    [set number, category name, metric name, score, lower_bound, upper_bound, sample_size]
    (the latter three are omitted for f-score results, since they don't apply there)
    """



    # figure out what to run
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
        if not do_this_category(bunch, set_name):
            continue
        print("\nEvaluating " + set_name)
        results.append([""]*7)
        for category_name in category_names:
            eval_class, info = category_name_to_set_class_and_metadata[category_name]
            if do_skip_category(info, use_testset, use_grapes, use_grapes_from_testset, use_grapes_from_ptb):
                # we can always try to find the appropriate subcorpus file...
                if predictions_directory is not None and info.subcorpus_filename is not None:
                    try:
                        # try to get the subcorpus from the same folder as the full corpus
                        print(f"Trying skipped category from single file {info.subcorpus_filename}.txt in {predictions_directory}")
                        results_here = run_single_file(eval_class, info, ".",
                                                       predictions_directory=predictions_directory,
                                                       do_error_analysis=do_error_analysis, parser_name=parser_name,
                                                       run_smatch=run_smatch)
                        rows = make_rows_for_results(category_name, filter_out_f1, filter_out_unlabeled_edge_attachment,
                                                     results_here, set_name)
                        results.extend(rows)
                    except Exception as e:
                        print(f"Can't get category {category_name}, error: {e}")
                        if fail_ok > -1:
                            results.append(make_empty_result(set_name, info.display_name))
                        else:
                            raise e
            else:

                if info.subcorpus_filename is None:  # testset
                    gold_graphs = gold_graphs_testset
                    predicted_graphs = predicted_graphs_testset
                else:
                    gold_graphs = gold_graphs_grapes
                    predicted_graphs = predicted_graphs_grapes

                evaluator = eval_class(gold_graphs, predicted_graphs, info, do_error_analysis=do_error_analysis,
                                       parser_name=parser_name, verbose_error_analysis=False, run_smatch=run_smatch)
                results_here = evaluate(evaluator, info, ".", predictions_directory=predictions_directory,
                                        fail_ok=fail_ok, run_smatch=run_smatch)

                rows = make_rows_for_results(category_name, filter_out_f1, filter_out_unlabeled_edge_attachment,
                                      results_here, set_name)
                results.extend(rows)
                if info.subtype == "structural_generalization":
                    by_size = evaluator.get_results_by_size()
                    struct_gen_by_size[info.display_name] = by_size

    return results, struct_gen_by_size


def make_rows_for_results(category_name, filter_out_f1, filter_out_unlabeled_edge_attachment, results_here,
                          set_name):
    rows = []
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
                rows.append([set_name[0], category_name_to_set_class_and_metadata[category_name][1].display_name, metric_name,
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
            rows.append([set_name[0], category_name_to_set_class_and_metadata[category_name][1].display_name, metric_name,
                            num_to_score(r[3]), "-", "-", "-"])
        else:
            print(
                "ERROR: Unexpected evaluation type! This means something unexpected went wrong (feel free to "
                "contact the developers of GrAPES for help, e.g. by filing an issue on GitHub).")
            print(r)
    return rows


def make_empty_result(set_name, category_name):
    return [set_name[0], category_name, "N/A", "N/A", "N/A", "N/A", "N/A"]


def do_skip_category(info, use_testset, use_grapes, use_grapes_from_testset, use_grapes_from_ptb):
    if not use_testset and is_testset_category(info):
        return True
    if not use_grapes and not is_testset_category(info):
        return True
    if not use_grapes_from_testset and is_grapes_category_with_testset_data(info):
        return True
    if not use_grapes_from_ptb and is_grapes_category_with_ptb_data(info):
        return True
    return False


def main():

    args = parse_args()
    if args.gold_amr_testset_file is not None and args.predicted_amr_testset_file is not None:
        gold_graphs_testset = load(args.gold_amr_testset_file, encoding="utf8")
        predicted_graphs_testset = load_predictions(args.predicted_amr_testset_file)
        if len(gold_graphs_testset) != len(predicted_graphs_testset):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the testset here."
                "Got " + str(len(gold_graphs_testset)) + " gold AMRs and " + str(len(predicted_graphs_testset))
                + " predicted AMRs.")
    else:
        gold_graphs_testset = predicted_graphs_testset = None

    if args.gold_amr_grapes_file is not None and args.predicted_amr_grapes_file is not None:
        gold_graphs_grapes = load(args.gold_amr_grapes_file, encoding="utf8")
        predicted_graphs_grapes = load_predictions(args.predicted_amr_grapes_file, encoding="utf8")
        predictions_directory = os.path.dirname(args.predicted_amr_grapes_file)

        if len(gold_graphs_grapes) != len(predicted_graphs_grapes):
            raise ValueError(
                "Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the grapes corpus here."
                "Got " + str(len(gold_graphs_grapes)) + " gold AMRs and " + str(len(predicted_graphs_grapes))
                + " predicted AMRs.")

    else:
        gold_graphs_grapes = predicted_graphs_grapes = predictions_directory = None

    instance_instructions = EvaluationInstanceInfo(
        fail_ok=-1 if args.strict else 0,
        do_error_analysis=args.error_analysis,
        parser_name=args.parser_name,
        run_smatch=args.smatch,
        print_f1_default=args.all_metrics,
        print_unlabeled_edge_attachment=args.all_metrics,

    )

    # run the evaluation
    results, by_size = get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes,
                          predictions_directory, args,
                          filter_out_f1=not args.all_metrics and not args.smatch, filter_out_unlabeled_edge_attachment=not args.all_metrics,
                                   bunch=args.bunch)

    store_results(args.parser_name, results)

    print_table = PrettyTable(field_names=["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"])
    print_table.align = "l"
    for row in results:
        print_table.add_row(row)

    if len(by_size) > 0:
        pretty_print_structural_generalisation_by_size(by_size)

    header = "\nAll results"
    if args.bunch is not None:
        header += f" for bunch {args.bunch}"
    print(header)
    print(print_table)


def store_results(parser_name, results, root_dir="."):
    results_dir = get_root_results_path(root_dir)
    os.makedirs(results_dir, exist_ok=True)
    if parser_name is not None:
        filename = parser_name
    else:
        filename = "results"
    csv.writer(open(f"{results_dir}/{filename}.csv", "w", encoding="utf8")).writerows(results)
    print(f"CSV of results written to {results_dir}/{filename}.csv")
    pickle.dump(results, open(f"{results_dir}/{filename}.pickle", "wb"))
    print(f"Pickle of results written to {results_dir}/results.pickle")


if __name__ == "__main__":
    main()

    # for key in category_name_to_set_class_and_metadata:
    #     if category_name_to_set_class_and_metadata[key][1].display_name != category_name_to_print_name[key]:
    #         print(key, category_name_to_set_class_and_metadata[key][1].display_name, category_name_to_print_name[key])
