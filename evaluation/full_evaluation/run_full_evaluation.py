import csv
import os
import pickle
import sys

from evaluation.util import num_to_score
from evaluation.corpus_metrics import compute_smatch_f_from_graph_lists
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata, is_copyrighted_data, \
    is_sanity_check
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from prettytable import PrettyTable
from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1, \
    CategoryEvaluation, STRUC_GEN, size_mappers, EVAL_TYPE_PRECISION
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
    is_testset_category, bunch2subcategory

args = sys.argv
if len(args) > 1:
    parser_names = args[1:]
else:
    # parser_names = ["amparser", "cailam", "amrbart"]
    parser_names = ["amparser"]

# update per use if desired
do_error_analysis = True
run_all_smatch = False
run_full_corpus_smatch = False

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
    gold_amrs, gold_grapes = import_gold_graphs()
    # results_path, pickle_path, by_size_pickle_path = make_results_path()
    parser_name2rows = dict()
    all_generalisations_by_size_dict = dict()

    for parser_name in parser_names:
        evaluation_instance_info = EvaluationInstanceInfo(
            root_dir="../..",
            run_smatch=run_all_smatch,
            subcorpus_predictions_directory_path_from_root=f"{path_to_parser_outputs}/{parser_name}-output",
            do_error_analysis=do_error_analysis,
            fail_ok=cat_fail_ok,
            verbose_error_analysis=False,
            # print_f1_default=True,  # not paid attention to by the table printer in this script
            # print_unlabeled_edge_attachment=True,
            parser_name=parser_name,
        )
        testset_parser_outs = load(evaluation_instance_info.default_testset_pred_file_path())
        grapes_parser_outs = load(evaluation_instance_info.full_grapes_pred_file_path())
        # os.makedirs(evaluation_instance_info.results_directory_path(), exist_ok=True)

        assert len(testset_parser_outs) == len(gold_amrs)
        assert len(grapes_parser_outs) == len(gold_grapes)
        print(f"{len(gold_grapes)} GrAPES graphs")
        print(f"{len(testset_parser_outs)} testset graphs")

        print("Running evaluation for", parser_name, "...")

        all_result_rows = []
        parser_name2rows[parser_name] = all_result_rows
        sums = []
        divisors = []

        if run_full_corpus_smatch:
            print("Running Smatch...")
            smatch = compute_smatch_f_from_graph_lists(gold_grapes, grapes_parser_outs)
            smatch_test = compute_smatch_f_from_graph_lists(gold_amrs, testset_parser_outs)
            print("Smatch done")
            rows = make_rows_for_results("Overall on novel GrAPES corpus", True, True,
                                  [[None, "Smatch", EVAL_TYPE_F1,  smatch[2], len(gold_grapes)]], "")
            all_result_rows.extend(rows)
            rows = make_rows_for_results("Overall on AMR 3.0 testset", True, True,
                                   [[None, "Smatch", EVAL_TYPE_F1,  smatch_test[2], len(gold_amrs)]], "")
            all_result_rows.extend(rows)


        generalisation_by_size = {}

        if run_all_smatch:
            print("We will run Smatch on all categories. This may take a while...\n"
                  " to avoid this, stop and change run_all_smatch to False.")
        for bunch in sorted(bunch2subcategory.keys()):
            sum_here = 0
            divisors_here = 0
            n, name = get_bunch_number_and_name(bunch)
            all_result_rows.append([n, name] + [""] * 5)
            # all_result_rows.append([bunch])
            print("Doing Bunch", bunch)

            for subcategory in bunch2subcategory[bunch]:
                eval_class, info = category_name_to_set_class_and_metadata[subcategory]
                evaluation_instance_info.given_single_file = False

                # get the appropriate corpora
                if is_testset_category(info):
                    gold = gold_amrs
                    pred = testset_parser_outs
                else:
                    gold = gold_grapes
                    pred = grapes_parser_outs

                evaluator = eval_class(gold, pred, info, evaluation_instance_info)

                # Structural generalisation results by size
                if info.subtype == STRUC_GEN and info.subcorpus_filename in size_mappers:
                    generalisation_by_size[info.display_name] =  evaluator.get_results_by_size()

                results_here = evaluate(evaluator, info, evaluation_instance_info)
                rows = make_rows_for_results(subcategory, evaluation_instance_info.print_f1(),
                                             evaluation_instance_info.print_unlabeled_edge_attachment, results_here, bunch)

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

                all_result_rows += rows
            sums.append(sum_here)
            divisors.append(divisors_here)

        print("\nRESULTS FOR", parser_name)

        # print("sums", sums, divisors)
        # for total, divisor in zip(sums, divisors):
        #     print(total / divisor)

        averages_table = PrettyTable(
            field_names=["Set", "Average"])
        averages_table.align = "l"
        for bunch, total, divisor in zip(bunch2subcategory.keys(), sums, divisors):
            averages_table.add_row([bunch, int((total / divisor)*100)])
        print(averages_table)


        results_path, pickle_path, by_size_pickle_path = make_results_path()
        if evaluation_instance_info.do_error_analysis:
            print("Error analysis pickles in", f"{root_dir_here}/error_analysis/{parser_name}/")
        table = pretty_print_structural_generalisation_by_size(generalisation_by_size)
        all_generalisations_by_size_dict[parser_name] = generalisation_by_size
        out_csv_by_size = f"{results_path}/{parser_name}_by_size.csv"
        csv.writer(open(out_csv_by_size, "w")).writerow(table.field_names)
        csv.writer(open(out_csv_by_size, "a", encoding="utf8")).writerows(table.rows)

        print("All result rows")

        print_table = PrettyTable(
            field_names=["Set", "Category", "Metric", "Score", "Lower bound", "Upper bound", "Sample size"])
        print_table.align = "l"
        for row in all_result_rows:
            print_table.add_row(row)
        print(print_table)

        # print(all_result_rows)
        # print_full_pretty_table(all_result_rows)
        csv_path = f"{results_path}/{parser_name}.csv"
        csv_rows = []
        for row in all_result_rows:
            if len(row) > 1:
                if row[0] is None:
                    row_to_append = [""]
                else:
                    row_to_append = [row[0]]
                row_to_append += row[1:]
                missing_entries = 7 - len(row)
                for i in range(missing_entries):
                    row_to_append.append("")
                csv_rows.append(row_to_append)
        csv.writer(open(csv_path, "w", encoding="utf8")).writerows(csv_rows)
        print("written to", csv_path)

    pickle.dump(parser_name2rows, open(pickle_path, "wb"))
    pickle.dump(all_generalisations_by_size_dict, open(by_size_pickle_path, "wb"))
    print("Results pickled in ", results_path)


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


# Broken. Use scripts/latex/csv2latex.py
#
# def make_latex_table(root_dir: str):
#     """
#     Might not matter: I think the csv2latex one works
#     """
#     result_rows_by_parser_name = pickle.load(open(root_dir + "/results_table.pickle", "rb"))
#
#     master_parser = parser_names[0]
#     master_rows = result_rows_by_parser_name[master_parser]
#
#     # results_rows_by_column = zip()
#     # print("\n###", list(results_rows_by_column))
#     # results_rows_by_column = zip(result_rows_by_parser_name["amparser"],
#     #                              # result_rows_by_parser_name["cailam"],
#     #                              # result_rows_by_parser_name["amrbart"]
#     #                              )
#
#     set_to_scores = dict()
#     current_scores = None
#
#     with open(root_dir + "/latex_results_table.txt", "w") as f:
#         set_id = ""
#         shade_row = True
#         for j, parser_row in enumerate(master_rows):
#             is_title_row = len(parser_row) == 1
#             if is_title_row:
#                 set_id = parser_row[0][0]
#                 current_scores = [[] for _ in range(len(parser_names))]
#                 set_to_scores[parser_row[0]] = current_scores
#                 continue
#
#             if parser_row[0] is None:
#                 dataset_name = ""
#             elif isinstance(parser_row[0], str):
#                 dataset_name = parser_row[0]
#             else:
#                 dataset_name = parser_row[0].get_latex_display_name()
#             # dataset_name = parser_rows[0][0]
#             metric_name = parser_row[1]
#
#             is_unlabeled_edge_row = metric_name == "Unlabeled edge recall"
#             is_smatch_row = "smatch" in metric_name.lower()
#             if is_unlabeled_edge_row or is_smatch_row:
#                 continue
#
#             is_sanity_check_row = "sanity" in dataset_name.lower()
#             is_prereq_row = "prereq" in metric_name.lower()
#             if set_id in ["1", "3", "4", "7"]:  # the sets that start a new table
#                 shade_row = True
#             else:
#                 if is_sanity_check_row or dataset_name.strip() == "":
#                     pass
#                 else:
#                     shade_row = not shade_row
#
#             shading_prefix = "\\rowcolor{lightlightlightgray}" if shade_row else ""
#             latex_line = f"\t\t{shading_prefix}{set_id} & {dataset_name} & {metric_name}"
#
#             is_success_rate_row = parser_row[2] == EVAL_TYPE_SUCCESS_RATE
#             if is_success_rate_row and not "precision" in metric_name.lower():
#                 for name in parser_names:
#                     assert parser_row[4] == result_rows_by_parser_name[name][4]
#             for i in range(len(parser_names)):
#                 if is_success_rate_row:
#                     wilson_ci = wilson_score_interval(row[3], row[4])
#                     score = num_to_score_with_preceding_0(row[3] / row[4])
#                     if not (is_prereq_row or is_sanity_check_row):
#                         current_scores[i].append(row[3] / row[4])
#                     if len(score) == 2:
#                         score = "\\phantom{1}" + score
#                     lower_bound = num_to_score_with_preceding_0(wilson_ci[0])
#                     upper_bound = num_to_score_with_preceding_0(wilson_ci[1])
#                     if len(lower_bound) == 2:
#                         lower_bound = lower_bound
#                     if len(upper_bound) == 2:
#                         upper_bound = "\\phantom{1}" + upper_bound
#                     # latex_line += f" & ${score}_{{{lower_bound}}}^{{{upper_bound}}}$"
#                     latex_line += f" & \\successScore{{{score}}}{{{lower_bound}}}{{{upper_bound}}}{{\\phantom{{1}}}}"
#                 else:
#                     if not (is_prereq_row or is_sanity_check_row):
#                         current_scores[i].append(row[3])
#                     latex_line += f" & {num_to_score_with_preceding_0(row[3])}\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ "
#
#             if is_success_rate_row:
#                 latex_line += f" & {parser_rows[0][4]}\\\\\n"
#             else:
#                 latex_line += f" & \\\\\n"
#
#             f.write(latex_line)
#
#             set_id = ""
#
#     with open(root_dir + "/latex_compact_table.txt", "w") as f:
#         for set_name, scores_by_parser in set_to_scores.items():
#             # print(set_name)
#             # print(scores_by_parser)
#             latex_line = f"\t\t{set_name}"
#             for scores in scores_by_parser:
#                 # average the scores
#                 score = sum(scores) / len(scores)
#                 latex_line += f" & {num_to_score_with_preceding_0(score)}"
#             latex_line += "\\\\\n"
#             f.write(latex_line)



    # 1 & Ambiguous
    # coreference & Recall & $6
    # _
    # {1} ^ {15}$ & & $39
    # _
    # {26} ^ {47}$ \todo
    # {put in actual
    # numbers} & ?\ \


# TODO: make a table that is average for categories (can't have wilson scores here,
#  because we DON'T want to normalize by sample size here)


def _get_row_evaluation_type(row):
    if len(row) >= 3:
        return row[2]
    else:
        return len(row)

def pretty_print_structural_generalisation_by_size(results):
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

    print("\nStructure generalisation results by size")
    print(table)
    return table

def make_rows_for_results(category_name, print_f1, print_unlabeled_edge_attachment, results_here,
                          set_name):
    rows = []
    for r in results_here:
        set_id = get_bunch_number_and_name(set_name)[0]
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
                rows.append([set_id, name, metric_name,
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
            rows.append([set_id, name, metric_name,
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

if __name__ == '__main__':
    main()