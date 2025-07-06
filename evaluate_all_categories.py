import argparse
import csv
import os
import pickle

from penman import load

from evaluate_single_category import SmartFormatter, load_predictions

from evaluation.full_evaluation.category_evaluation.subcategory_info import is_grapes_category_with_testset_data, \
    is_grapes_category_with_ptb_data
from evaluation.full_evaluation.run_full_evaluation import run_single_file, evaluate, \
    pretty_print_structural_generalisation_by_size, make_rows_for_results, get_bunch_number_and_name

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

def instance_info_from_args(args):
    instance_instructions = EvaluationInstanceInfo(
        path_to_full_testset_predictions_file_from_root=args.predicted_amr_testset_file,
        path_to_grapes_predictions_file_from_root=args.predicted_amr_grapes_file,
        path_to_gold_testset_file_from_root=args.gold_amr_testset_file,
        do_error_analysis=args.error_analysis,
        parser_name=args.parser_name,
        run_smatch=args.smatch,
        fail_ok=-1 if args.strict else 0,
        print_f1_default=args.all_metrics,
        print_unlabeled_edge_attachment=args.all_metrics,
    )
    return instance_instructions

def do_this_category(bunch, category_name):
    return bunch is None or category_name.startswith(str(bunch)+".")


def get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes,
                instance_info: EvaluationInstanceInfo, bunch=None):
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
        # n, name = get_bunch_number_and_name(set_name)
        # results.append([n, name] + [""]*5)
        for category_name in category_names:
            eval_class, info = category_name_to_set_class_and_metadata[category_name]
            instance_info.given_single_file = False
            if do_skip_category(info, use_testset, use_grapes, use_grapes_from_testset, use_grapes_from_ptb):
                # we can always try to find the appropriate subcorpus file...
                if instance_info.predictions_directory_path() is not None and info.subcorpus_filename is not None:
                    try:
                        # try to get the subcorpus from the same folder as the full corpus
                        print(f"Trying skipped category from single file {info.subcorpus_filename}.txt in"
                              f" {instance_info.predictions_directory_path()}")
                        results_here = run_single_file(eval_class, info, instance_info)
                        rows = make_rows_for_results(category_name, instance_info.print_f1(),
                                                     instance_info.print_unlabeled_edge_attachment,
                                                     results_here, set_name)
                        results.extend(rows)
                    except Exception as e:
                        print(f"Can't get category {category_name}, error: {e}")
                        if instance_info.fail_ok > -1:
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

                evaluator = eval_class(gold_graphs, predicted_graphs, info, instance_info)
                results_here = evaluate(evaluator, info, instance_info)

                rows = make_rows_for_results(category_name, instance_info.print_f1(),
                                             instance_info.print_unlabeled_edge_attachment, results_here, set_name)
                results.extend(rows)
                if info.subtype == STRUC_GEN:
                    by_size = evaluator.get_results_by_size()
                    struct_gen_by_size[info.display_name] = by_size

    return results, struct_gen_by_size


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

    # etract info from command line
    args = parse_args()
    instance_info = instance_info_from_args(args)

    # read in the graphs from the given paths
    if instance_info.gold_testset_path() is not None and instance_info.testset_pred_file_path() is not None:
    #if args.gold_amr_testset_file is not None and args.predicted_amr_testset_file is not None:
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

    # run the evaluation
    results, by_size = get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset,
                                   predicted_graphs_grapes, instance_info, bunch=args.bunch)

    store_results(results, instance_info)

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


def store_results(results, instance_info: EvaluationInstanceInfo):
    results_dir = f"{instance_info.root_dir}/data/processed/results/from_evaluate_all_categories"
    os.makedirs(results_dir, exist_ok=True)
    filename = instance_info.parser_name
    out_file = f"{results_dir}/{filename}.csv"
    csv.writer(open(out_file, "w", encoding="utf8")).writerows(results)
    print(f"CSV of results written to {out_file}")
    out_file = f"{results_dir}/{filename}.pickle"
    pickle.dump(results, open(out_file, "wb"))
    print(f"Pickle of results written to {out_file}")


if __name__ == "__main__":
    main()

    # for key in category_name_to_set_class_and_metadata:
    #     if category_name_to_set_class_and_metadata[key][1].display_name != category_name_to_print_name[key]:
    #         print(key, category_name_to_set_class_and_metadata[key][1].display_name, category_name_to_print_name[key])
