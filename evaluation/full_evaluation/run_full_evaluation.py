import csv
import os
import pickle
import sys

from evaluation.util import num_to_score
from evaluation.corpus_metrics import compute_smatch_f_from_graph_lists
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata, is_copyrighted_data, \
    is_sanity_check, is_grapes_category_with_testset_data, is_grapes_category_with_ptb_data
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from prettytable import PrettyTable
from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1, \
    CategoryEvaluation, STRUC_GEN, size_mappers, EVAL_TYPE_PRECISION
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
    is_testset_category, bunch2subcategory, get_bunch_name_for_number, \
    get_bunch_categories_for_number, get_bunch_display_name_for_number

args = sys.argv
if len(args) > 1:
    parser_names = args[1:]
else:
    # parser_names = ["amparser", "cailam", "amrbart"]
    parser_names = ["amparser"]

# update per use if desired
do_error_analysis = True
run_all_smatch = True
run_full_corpus_smatch = True

# ERROR HANDLING GLOBAL
# raise an error if any category doesn't work
# cat_fail_ok = -1
# raise an error if any category not in the copyrighted data raises an error
# and if a category in the copyrighted data raises an AssertionError, try loading the individual files
# if that fails, make an empty row
cat_fail_ok = 0
# just skip anything that doesn't work and make an empty row for it
# cat_fail_ok = 1

# globals
root_dir_here = "../.."

# update for your setup
path_to_parser_outputs = "data/processed/parser_outputs"
# full_grapes_name = "full_corpus"
gold_testset_path = f"{root_dir_here}/data/raw/gold/test.txt"


def get_root_results_path(root_dir):
    x = f"{root_dir}/data/processed/results"
    return x


def make_results_path():
    results_path = f"{get_root_results_path(root_dir_here)}/from_run_full_evaluation"
    os.makedirs(results_path, exist_ok=True)
    pickle_path = f"{results_path}/all_parsers_results_table.pickle"
    by_size_pickle_path = f"{results_path}/all_parsers_structural_generation_by_size.pickle"
    return results_path, pickle_path, by_size_pickle_path


def import_gold_graphs():
    gold_amrs = load(gold_testset_path)

    gold_grapes = load(f"{root_dir_here}/corpus/corpus.txt")
    return gold_amrs, gold_grapes

def get_bunch_number_and_name(bunch_string):
    """
    Structure: n. name of bunch
    """
    parts = bunch_string.split(". ")
    name = ".".join(parts[1:])  # put it back together just in case
    return parts[0], name


def create_results_pickles():
    """
    Main evaluation function. Runs evaluation on all categories for all parsers. See/update global variables
     for where parser results should be stored.

    Pickles and prints the results
    """
    gold_testset, gold_grapes = import_gold_graphs()
    # results_path, pickle_path, by_size_pickle_path = make_results_path()
    parser_name2rows = dict()
    all_generalisations_by_size_dict = dict()

    for parser_name in parser_names:

        instance_info = EvaluationInstanceInfo(
            root_dir="../..",
            run_smatch=run_all_smatch,
            run_full_corpus_smatch=run_full_corpus_smatch,
            run_structural_generalisation_smatch=True,
            subcorpus_predictions_directory_path_from_root=f"{path_to_parser_outputs}/{parser_name}-output",
            do_error_analysis=do_error_analysis,
            fail_ok=cat_fail_ok,
            verbose_error_analysis=False,
            # print_f1_default=True,  # not paid attention to by the table printer in this script
            # print_unlabeled_edge_attachment=True,
            parser_name=parser_name,
        )
        try:
            testset_parser_outs = load_predictions(instance_info.default_testset_pred_file_path())
            grapes_parser_outs = load_predictions(instance_info.full_grapes_pred_file_path())
            # os.makedirs(evaluation_instance_info.results_directory_path(), exist_ok=True)
        except FileNotFoundError as e:
            print(f"\n!! {parser_name} outputs not found. Check your parser outputs are in data/processed/parser_outputs/<parser_name>-output/ and named full_corpus.txt (for GrAPES) and testset.txt (for the original AMR 3.0 dataset)\n")
            raise(e)

        assert len(testset_parser_outs) == len(gold_testset)
        assert len(grapes_parser_outs) == len(gold_grapes)
        print(f"{len(gold_grapes)} GrAPES graphs")
        print(f"{len(testset_parser_outs)} testset graphs")

        print("Running evaluation for", parser_name, "...")
        if instance_info.run_smatch:
            print("We will run Smatch on all categories. This may take a while...\n"
                  " to avoid this, stop and change run_all_smatch to False.")


        results, by_size, sums, divisors, dont_print_these_averages = get_results(gold_testset, gold_grapes, testset_parser_outs, grapes_parser_outs, instance_info)
        parser_name2rows[parser_name] = results
        store_results(results, instance_info,
                      results_dir=f"{instance_info.root_dir}/data/processed/results/from_run_full_evaluation")

        print("\nRESULTS FOR", parser_name)

        display_results(results)

        results_path, pickle_path, by_size_pickle_path = make_results_path()

        display_and_store_averages(divisors, parser_name, results_path, sums, dont_print_these_averages)

        if instance_info.do_error_analysis:
            print("Error analysis pickles in", f"{root_dir_here}/error_analysis/{parser_name}/")

        if len(by_size) > 0:
            display_and_store_by_size(by_size, parser_name, results_path, all_generalisations_by_size_dict)

        # TODO Had two versions, and this one might still contain stuff I forgot to add so keeping it for now
        # all_result_rows = []
        # parser_name2rows[parser_name] = all_result_rows
        # sums = []
        # divisors = []
        #
        # if run_full_corpus_smatch:
        #     print("Running Smatch...")
        #     smatch = compute_smatch_f_from_graph_lists(gold_grapes, grapes_parser_outs)
        #     smatch_test = compute_smatch_f_from_graph_lists(gold_amrs, testset_parser_outs)
        #     print("Smatch done")
        #     rows = make_rows_for_results("Overall on novel GrAPES corpus", True, True,
        #                           [[None, "Smatch", EVAL_TYPE_F1,  smatch[2], len(gold_grapes)]], "")
        #     all_result_rows.extend(rows)
        #     rows = make_rows_for_results("Overall on AMR 3.0 testset", True, True,
        #                            [[None, "Smatch", EVAL_TYPE_F1,  smatch_test[2], len(gold_amrs)]], "")
        #     all_result_rows.extend(rows)
        #
        #
        # generalisation_by_size = {}
        #
        # if run_all_smatch:
        #     print("We will run Smatch on all categories. This may take a while...\n"
        #           " to avoid this, stop and change run_all_smatch to False.")
        # for bunch in sorted(bunch2subcategory.keys()):
        #     sum_here = 0
        #     divisors_here = 0
        #     # n, name = get_bunch_number_and_name(bunch)
        #     # all_result_rows.append([n, name] + [""] * 5)
        #     # all_result_rows.append([bunch])
        #     print("Doing Bunch", bunch)
        #
        #     for subcategory in bunch2subcategory[bunch]:
        #         eval_class, info = category_name_to_set_class_and_metadata[subcategory]
        #         evaluation_instance_info.given_single_file = False
        #
        #         # get the appropriate corpora
        #         if is_testset_category(info):
        #             gold = gold_amrs
        #             pred = testset_parser_outs
        #         else:
        #             gold = gold_grapes
        #             pred = grapes_parser_outs
        #
        #         evaluator = eval_class(gold, pred, info, evaluation_instance_info)
        #
        #         # Structural generalisation results by size
        #         if info.subtype == STRUC_GEN and info.subcorpus_filename in size_mappers:
        #             generalisation_by_size[info.display_name] =  evaluator.get_results_by_size()
        #
        #         results_here = evaluate(evaluator, info, evaluation_instance_info)
        #         rows = make_rows_for_results(subcategory, evaluation_instance_info.print_f1(),
        #                                      evaluation_instance_info.print_unlabeled_edge_attachment, results_here, bunch)
        #
        #         for r in results_here:
        #             metric_name = r[1]
        #
        #             is_sanity_check_row = is_sanity_check(info)
        #             is_prereq_row = "prereq" in metric_name.lower()
        #             is_smatch_row = "smatch" in metric_name.lower()
        #             is_unlabelled_row = "unlabel" in metric_name.lower()
        #             exclude_from_average = is_sanity_check_row or is_prereq_row or is_smatch_row or is_unlabelled_row
        #             if not exclude_from_average:
        #                 sum_here += r[3] / r[4]
        #                 divisors_here += 1
        #
        #         all_result_rows += rows
        #     sums.append(sum_here)
        #     divisors.append(divisors_here)
        #
        # print("\nRESULTS FOR", parser_name)
        # results_path, pickle_path, by_size_pickle_path = make_results_path()
        #
        # # print("sums", sums, divisors)
        # # for total, divisor in zip(sums, divisors):
        # #     print(total / divisor)
        #
        # averages_table = PrettyTable(
        #     field_names=["Set", "Average"])
        # averages_table.align = "l"
        # for bunch, total, divisor in zip(bunch2subcategory.keys(), sums, divisors):
        #     averages_table.add_row([bunch, int((total / divisor)*100)])
        # print(averages_table)
        # with open(f"{results_path}/{parser_name}_averages.csv", "w") as f:
        #     csv.writer(f).writerows(averages_table.rows)
        #
        #
        # if evaluation_instance_info.do_error_analysis:
        #     print("Error analysis pickles in", f"{root_dir_here}/error_analysis/{parser_name}/")
        # table = pretty_print_structural_generalisation_by_size(generalisation_by_size)
        # all_generalisations_by_size_dict[parser_name] = generalisation_by_size
        # out_csv_by_size = f"{results_path}/{parser_name}_by_size.csv"
        # csv.writer(open(out_csv_by_size, "w")).writerow(table.field_names)
        # csv.writer(open(out_csv_by_size, "a", encoding="utf8")).writerows(table.rows)
        #
        # print("All result rows")
        #
        # print_table = PrettyTable(
        #     field_names=["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"])
        # print_table.align = "l"
        # current_set = 0
        # for i, row in enumerate(all_result_rows):
        #     previous_set = current_set
        #     current_set = int(row[0])
        #     if current_set != previous_set:
        #         total = sums[current_set - 1]
        #         divisor = divisors[current_set - 1]
        #         print_table.add_row([row[0], bunch_number2name[current_set], "Average", int((total / divisor)*100), "", "", ""])
        #     try:
        #         print_divider = int(all_result_rows[i + 1][0]) > current_set
        #     except IndexError:
        #         print_divider = False
        #     print_table.add_row(row, divider=print_divider)
        # print(print_table)
        #
        # # print(all_result_rows)
        # # print_full_pretty_table(all_result_rows)
        # csv_path = f"{results_path}/{parser_name}.csv"
        # csv_rows = [["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"]]
        # for row in all_result_rows:
        #     if len(row) > 1:
        #         if row[0] is None:
        #             row_to_append = [""]
        #         else:
        #             row_to_append = [row[0]]
        #         row_to_append += row[1:]
        #         missing_entries = 7 - len(row)
        #         for i in range(missing_entries):
        #             row_to_append.append("")
        #         csv_rows.append(row_to_append)
        # csv.writer(open(csv_path, "w", encoding="utf8")).writerows(csv_rows)
        # print("written to", csv_path)

    pickle.dump(parser_name2rows, open(pickle_path, "wb"))
    pickle.dump(all_generalisations_by_size_dict, open(by_size_pickle_path, "wb"))
    print("Results pickled in ", results_path)


def display_and_store_by_size(by_size, parser_name, results_path, all_generalisations_by_size_dict=None):
    table = structural_generalisation_by_size_as_table(by_size)
    if all_generalisations_by_size_dict is not None:
        all_generalisations_by_size_dict[parser_name] = by_size
    out_csv_by_size = f"{results_path}/{parser_name}_by_size.csv"
    csv.writer(open(out_csv_by_size, "w")).writerow(table.field_names)
    csv.writer(open(out_csv_by_size, "a", encoding="utf8")).writerows(table.rows)
    print(f"Wrote structural generalisation results by size to {out_csv_by_size}")
    print("\nStructure generalisation results by size")
    print(table)
    return all_generalisations_by_size_dict


def display_and_store_averages(divisors, parser_name, results_path, sums, dont_print_these_averages, only_bunch=None):
    sums_and_divisors = list(zip(sums, divisors))
    averages_table = PrettyTable(
        field_names=["Set", "Average"])
    averages_table.align = "l"
    index_in_results = 0
    if only_bunch is not None:
        total, divisor = sums_and_divisors[index_in_results]
        if divisor > 0:
            averages_table.add_row([get_bunch_display_name_for_number(only_bunch), int((total / divisor) * 100)])
        else:
            averages_table.add_row([get_bunch_display_name_for_number(only_bunch), "-"])
    else:
        for bunch in range(len(bunch2subcategory)):
            bunch_number = bunch + 1
            if bunch_number not in dont_print_these_averages:
                total, divisor = sums_and_divisors[index_in_results]
                if divisor > 0:
                    averages_table.add_row([get_bunch_display_name_for_number(bunch_number), int((total / divisor) * 100)])
                else:
                    averages_table.add_row([get_bunch_display_name_for_number(bunch_number), "-"])
                index_in_results += 1
            else:
                averages_table.add_row([get_bunch_display_name_for_number(bunch_number), "-"])
    print(averages_table)
    with open(f"{results_path}/{parser_name}_averages.csv", "w") as f:
        csv.writer(f).writerows(averages_table.rows)
    print(f"Wrote averages to {results_path}/{parser_name}_averages.csv")


def evaluate(evaluator: CategoryEvaluation, info: SubcategoryMetadata, instance_info: EvaluationInstanceInfo):
    """
    Runs the given evaluator.
    If it fails, tries on individual files.
    Args:
        evaluator: initialised CategoryEvaluation class
        info: SubcategoryMetadata about this category
        instance_info: information about this evaluation instance, like parser name
    Returns:
        list of rows of results: [dataset name, metric_name, eval_type] + metric_results
    """
    if instance_info.fail_ok == -1:
        try:
            rows = evaluator.run_evaluation()
            return rows
        except Exception as e:
            print("ERROR in dataset", info.display_name, info.name, file=sys.stderr)
            raise e
    try:
        rows = evaluator.run_evaluation()
        assert len(rows) > 0, "No results!"
        return rows
    except AssertionError as e:
        print("WARNING: error trying to process", info.subcorpus_filename, e, file=sys.stderr)

        if is_copyrighted_data(info) :
            try:
                print("Copyrighted data may not be in parser outputs. Trying with individual files.",
                          file=sys.stderr)
                rows = run_single_file(type(evaluator), info, instance_info)
                print("OK", file=sys.stderr)
                return rows

            except Exception as e:
                return warn_and_make_empty_row(e, info)
                # raise e
        else:
            if instance_info.fail_ok == 1:
                return warn_and_make_empty_row(e, info)
            raise e
    except Exception as e:
        if instance_info.fail_ok == 1:
            return warn_and_make_empty_row(e, info)
        raise e


def warn_and_make_empty_row(e, info):
    print("Couldn't process", info.display_name, e, file=sys.stderr)
    return [CategoryEvaluation.make_empty_row(category_name=info.display_name)]


def run_single_file(eval_class, info: SubcategoryMetadata, instance_info: EvaluationInstanceInfo):
    """
    Evaluates one subcorpus file
    Args:
        instance_info: EvaluationInstanceInfo containing info about this run of the evaluation for this parser
        eval_class: CategoryEvaluation class to initialise
        info: SubcategoryMetadata for the file
    Returns: rows of results (list of lists)
    """
    if info.subcorpus_filename is None:
        raise ValueError(f"{info.display_name} is an AMR 3.0 category, but we're trying to get it from a GrAPES corpus file")
    pred = load(f"{instance_info.predictions_directory_path()}/{info.subcorpus_filename}.txt")
    gold = load(f"{instance_info.root_dir}/corpus/subcorpora/{info.subcorpus_filename}.txt")
    instance_info.given_single_file = True
    evaluator = eval_class(gold, pred, info, instance_info)
    rows = evaluator.run_evaluation()
    return rows


# def print_full_pretty_table(result_rows):
#     table = PrettyTable()
#     table.field_names = ["Dataset", "Metric", "Score", "Wilson CI", "Sample size"]
#     table.align = "l"
#     for row in result_rows:
#         if row[0] is None:
#             category = ""
#         elif isinstance(row[0], str):
#             category = row[0]
#         else:
#             category = row[0].display_name
#         eval_type = _get_row_evaluation_type(row)
#         if eval_type in [EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_PRECISION]:
#             wilson_ci = wilson_score_interval(row[3], row[4])
#             if row[4] > 0:
#                 # total predictions varies by parser, so don't print total here
#                 total_print = "-" if eval_type==EVAL_TYPE_PRECISION else row[4]
#                 table.add_row([category, row[1], num_to_score_with_preceding_0(row[3] / row[4]),
#                                f"[{num_to_score_with_preceding_0(wilson_ci[0])}, {num_to_score_with_preceding_0(wilson_ci[1])}]",
#                                total_print])
#             else:
#                 print("Division by zero!", file=sys.stderr)
#                 print(row[0], row[1:], file=sys.stderr)
#
#         elif eval_type == EVAL_TYPE_F1:
#             if len(row) > 4:
#                 sample_size = row[4]
#             else:
#                 sample_size = ""
#             table.add_row([category, row[1], num_to_score_with_preceding_0(row[3]), "", sample_size])
#         elif eval_type == EVAL_TYPE_NONE:
#             table.add_row(["", "", "", "", ""])
#             table.add_row([category, "", "", "", ""])
#         elif eval_type == EVAL_TYPE_NA:
#             table.add_row([category, row[1], "N/A", "N/A", "N/A"])
#         else:
#             print(row)
#             raise Exception(f"Unknown evaluation type: {eval_type}")
#     print(table)




def _get_row_evaluation_type(row):
    if len(row) >= 3:
        return row[2]
    else:
        return len(row)

def structural_generalisation_by_size_as_table(results):
    """
    Prints the structural generalisation results split up by size
    Args:
        results: dict from parser name to dataset name to dict from size to score
    Return: PrettyTable
    """
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
                row.append(int(results[dataset][n] * 100))
            else:
                row.append("")
        table.add_row(row)
    return table

def make_rows_for_results(category_name, print_f1, print_unlabeled_edge_attachment, results_here,
                          set_id, set_name):
    rows = []
    for r in results_here:
        try:
            name = category_name_to_set_class_and_metadata[category_name][1].display_name
        except KeyError:
            name = category_name
        metric_name = r[1]
        if not print_f1 and metric_name == "Smatch":
            continue
        if not print_unlabeled_edge_attachment and metric_name == "Unlabeled edge recall":
            continue
        metric_type = r[2]
        if metric_type in [EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_PRECISION]:
            wilson_ci = wilson_score_interval(r[3], r[4])
            if r[4] > 0:
                rows.append([set_id, set_name, name, metric_name,
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
            rows.append([set_id, set_name, name, metric_name,
                            num_to_score(r[3]), "-", "-", r[4]])
        else:
            print(
                "ERROR: Unexpected evaluation type! This means something unexpected went wrong (feel free to "
                "contact the developers of GrAPES for help, e.g. by filing an issue on GitHub).")
            print(metric_type)
    return rows




def main():
    create_results_pickles()
    # make_latex_table(results_path)

def display_results(results, bunch=None):
    print_table = PrettyTable(
        field_names=["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"])
    print_table.align = "l"
    for row in results:
        print_table.add_row([row[0]] + row[2:])
    header = "\nAll results"
    if bunch is not None:
        header += f" for bunch {bunch}"
    print(header)
    print(print_table)


def store_results(results, instance_info: EvaluationInstanceInfo, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    filename = instance_info.parser_name
    out_file = f"{results_dir}/{filename}.csv"
    csv_rows = [["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"]] + results
    csv.writer(open(out_file, "w", encoding="utf8")).writerows(csv_rows)
    print(f"CSV of results written to {out_file}")
    out_file = f"{results_dir}/{filename}.pickle"
    pickle.dump(results, open(out_file, "wb"))



def get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes,
                instance_info: EvaluationInstanceInfo,
                use_grapes=True, use_grapes_from_ptb=True, use_grapes_from_testset=True, use_testset=True, bunch=None):
    """
    Creates a list of result rows. Each row has the following format:
    [set number, category name, metric name, score, lower_bound, upper_bound, sample_size]
    (the latter three are omitted for f-score results, since they don't apply there)

    plus additional metrics
    Return results, struct_gen_by_size, sums, divisors
    """
    results = []
    struct_gen_by_size = {}
    sums = []
    divisors = []
    dont_print_these_averages = []

    # Smatch on the full corpora
    if instance_info.run_full_corpus_smatch:
        print("Running Smatch...")
        if use_grapes:
            smatch = compute_smatch_f_from_graph_lists(gold_graphs_grapes, predicted_graphs_grapes)
            rows = make_rows_for_results("Overall on novel GrAPES corpus", True, True,
                                         [[None, "Smatch", EVAL_TYPE_F1, smatch[2], len(gold_graphs_grapes)]], 0, "")
            results.extend(rows)
        if use_testset:
            smatch_test = compute_smatch_f_from_graph_lists(gold_graphs_testset, predicted_graphs_testset)
            rows = make_rows_for_results("Overall on AMR 3.0 testset", True, True,
                                         [[None, "Smatch", EVAL_TYPE_F1, smatch_test[2], len(gold_graphs_testset)]], 0, "")
            results.extend(rows)
        print("Smatch done")


    # loop through bunches
    for i in range(1, len(bunch2subcategory)+1):
        sum_here = 0  # for bunch averages
        divisors_here = 0 # "
        set_name = get_bunch_name_for_number(i)
        display_set_name = f"{i}. {set_name}"
        category_names = get_bunch_categories_for_number(i)
        if not do_this_category(bunch, i):
            continue
        print(f"\nEvaluating {display_set_name}")
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
                        for r in results_here:
                            metric_name = r[1]

                            is_sanity_check_row = is_sanity_check(info)
                            is_prereq_row = "prereq" in metric_name.lower()
                            is_smatch_row = "smatch" in metric_name.lower()
                            is_unlabelled_row = "unlabel" in metric_name.lower()
                            exclude_from_average = is_sanity_check_row or is_prereq_row or is_smatch_row or is_unlabelled_row
                            if not exclude_from_average:
                                sum_here += r[3] / r[4]
                                divisors_here += 1
                        rows = make_rows_for_results(category_name, instance_info.print_f1(),
                                                     instance_info.print_unlabeled_edge_attachment,
                                                     results_here, i, set_name)
                        results.extend(rows)
                    except Exception as e:
                        print(f"Can't get category {category_name}, error: {e}")
                        if instance_info.fail_ok > -1:
                            results.append(make_empty_result(i, set_name, info.display_name))
                            dont_print_these_averages.append(i)
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

                for r in results_here:
                    metric_name = r[1]

                    is_sanity_check_row = is_sanity_check(info)
                    is_prereq_row = "prereq" in metric_name.lower()
                    is_smatch_row = "smatch" in metric_name.lower()
                    is_unlabelled_row = "unlabel" in metric_name.lower()
                    exclude_from_average = is_sanity_check_row or is_prereq_row or is_smatch_row or is_unlabelled_row
                    if not exclude_from_average:
                        sum_here += r[3] / r[4]
                        divisors_here += 1

                rows = make_rows_for_results(category_name, instance_info.print_f1(),
                                             instance_info.print_unlabeled_edge_attachment, results_here, i, set_name)
                results.extend(rows)
                if info.subtype == STRUC_GEN and info.subcorpus_filename in size_mappers:
                    by_size = evaluator.get_results_by_size()
                    struct_gen_by_size[info.display_name] = by_size


        sums.append(sum_here)
        divisors.append(divisors_here)

    return results, struct_gen_by_size, sums, divisors, dont_print_these_averages


def make_empty_result(set_id, set_name, category_name):
    return [set_id, set_name, category_name, "N/A", "N/A", "N/A", "N/A", "N/A"]


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


def load_predictions(predictions_path, encoding="utf8"):
    """
    Add some printing around loading predictions in case of warnings from Penman
    """
    print("\nLoading predicted AMRs...")
    predicted_amrs = load(predictions_path, encoding=encoding)
    print("Done\n")
    return predicted_amrs


def do_this_category(bunch_to_do, bunch_id):
    return bunch_to_do is None or bunch_to_do == bunch_id


if __name__ == '__main__':
    main()