import csv
import os
import pickle
import sys

from evaluation.corpus_metrics import compute_smatch_f_from_graph_lists
from evaluation.full_evaluation.category_evaluation.evaluation_classes import PPAttachmentAlone
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.novel_corpus.structural_generalization import size_mappers
from evaluation.util import num_to_score_with_preceding_0
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from prettytable import PrettyTable
from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1, \
    CategoryEvaluation
from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, bunch2subcategory, \
    is_copyrighted_data, is_testset_category

# globals
root_dir_here = "../../"
path_to_parser_outputs = f"{root_dir_here}/data/raw/parser_outputs/"

# parser_names = ["amparser", "cailam", "amrbart"]
parser_names = ["amparser"]
full_grapes_name = "full_corpus"
gold_testset_path = f"{root_dir_here}/data/raw/gold/test.txt"

def get_results_path(root_dir):
    x = f"{root_dir}/data/processed/results"
    os.makedirs(x, exist_ok=True)
    return x

results_path = f"{get_results_path(root_dir_here)}/from_run_all_evaluations"
os.makedirs(results_path, exist_ok=True)

pickle_path = f"{results_path}/all_parsers_results_table.pickle"
by_size_pickle_path = f"{results_path}/all_parsers_structural_generation_by_size.pickle"

def import_graphs():
    gold_amrs = load(gold_testset_path)

    gold_grapes = load(f"{root_dir_here}/corpus/corpus.txt")
    return gold_amrs, gold_grapes


def get_predictions_path_for_parser(parser):
    return f"{path_to_parser_outputs}/{parser}-output"


def load_parser_output(subcorpus_name, root_dir=root_dir_here, parser_name=None, predictions_directory=None):
    """
    Use the predictions directory, or make it from the parser name, to load a subcorpus.
    Args:
        subcorpus_name:
        root_dir: path to root of the project from wherever this is being called
        parser_name: Optional
        predictions_directory: Optional if get_predictions_path_for_parser can get the path right

    Returns: list of Penman graphs
    """
    if parser_name is not None:
        try:
            return load(f"{get_predictions_path_for_parser(parser_name)}/{subcorpus_name}.txt")
        except FileNotFoundError:
            pass
    if predictions_directory is not None:
        graphs = load(f"{root_dir}/{predictions_directory}/{subcorpus_name}.txt")
        if len(graphs) == 0:
            print(f"No graphs found for {subcorpus_name}", file=sys.stderr)
        return graphs
    else:
        raise ValueError("parser name or predictions directory must be specified")


def create_results_pickles():
    """
    Main evaluation function. Runs evaluation on all categories for all parsers. See/update global variables
     for where parser results should be stored.

    Pickles and prints the results
    """
    gold_amrs, gold_grapes = import_graphs()
    parser_name2rows = dict()
    all_generalisations_by_size_dict = dict()

    for parser_name in parser_names:
        testset_parser_outs = load_parser_output("testset", parser_name=parser_name)
        grapes_parser_outs = load_parser_output(full_grapes_name, parser_name=parser_name)

        print("Running", parser_name, "...")

        smatch = compute_smatch_f_from_graph_lists(gold_grapes, grapes_parser_outs)
        print("Smatch on GrAPES:")
        print(smatch)

        all_result_rows = []
        parser_name2rows[parser_name] = all_result_rows
        all_result_rows.append(["Overall on novel GrAPES corpus", "Smatch", EVAL_TYPE_F1,  smatch[2], len(gold_grapes)])

        generalisation_by_size = {}

        for bunch in sorted(bunch2subcategory.keys()):

            all_result_rows.append([bunch])
            print("Doing Bunch", bunch)

            for subcategory in bunch2subcategory[bunch]:
                eval_class, info = category_name_to_set_class_and_metadata[subcategory]

                # get the appropriate corpora
                if is_testset_category(info):
                    gold = gold_amrs
                    pred = testset_parser_outs
                else:
                    gold = gold_grapes
                    pred = grapes_parser_outs

                evaluator = eval_class(gold, pred, root_dir_here, info)

                # Structural generalisation results by size
                if info.subtype == "structural_generalization" and info.subcorpus_filename in size_mappers:
                    generalisation_by_size[info.display_name] =  evaluator.get_results_by_size()

                rows = evaluate(evaluator, info, root_dir_here, parser_name, None)
                all_result_rows += rows

        print("\nRESULTS FOR", parser_name)
        pretty_print_structural_generalisation_by_size(generalisation_by_size)
        all_generalisations_by_size_dict[parser_name] = generalisation_by_size

        print("All result rows")
        # print(all_result_rows)
        print_pretty_table(all_result_rows)
        csv.writer(open(f"{results_path}/{parser_name}.csv", "w", encoding="utf8")).writerows(all_result_rows)

    pickle.dump(parser_name2rows, open(pickle_path, "wb"))
    pickle.dump(all_generalisations_by_size_dict, open(by_size_pickle_path, "wb"))
    print("Results pickled in ", results_path)


def evaluate(evaluator: CategoryEvaluation, info: SubcategoryMetadata, root_dir=root_dir_here, parser_name=None, predictions_directory=None):
    """
    Runs the given evaluator.
    If it fails, tries on individual files.
    Args:
        evaluator: initialised CategoryEvaluation class
        info: SubcategoryMetadata about this category
        root_dir:
        parser_name: Can be used to get predictions if load_parser_output works, or use predictions_directory
        predictions_directory:

    Returns:
        list of rows of results: [dataset name, metric_name, eval_type] + metric_results
    """
    try:
        rows = evaluator.run_evaluation()
        return rows
    except AssertionError as e:
        print("WARNING: error trying to process", info.subcorpus_filename, e, file=sys.stderr)

        if is_copyrighted_data(info) :
            try:
                print("Copyrighted data may not be in parser outputs. Trying with individual files.",
                          file=sys.stderr)
                rows = run_single_file(type(evaluator), info, root_dir=root_dir, parser_name=parser_name,
                                       predictions_directory=predictions_directory)
                print("OK", file=sys.stderr)
                return rows

            except Exception as e:
                print("Couldn't process", info.subcorpus_filename, e, file=sys.stderr)
                # raise e
        elif info.subcorpus_filename == "pp_attachment":
            try:
                print("Trying PP attachment files", file=sys.stderr)
                evaluator = PPAttachmentAlone(root_dir, info, predictions_directory)
                rows = evaluator.run_evaluation()
                print("OK", file=sys.stderr)
                return rows
            except Exception as e:
                print("Couldn't process", info.display_name, e, file=sys.stderr)
                # raise e
        else:
            raise e


def run_single_file(eval_class, info: SubcategoryMetadata, root_dir=root_dir_here, parser_name=None, predictions_directory=None):
    """
    Evaluates one subcorpus file
    Args:
        eval_class: CategoryEvaluation class to initialise
        info: SubcategoryMetadata for the file
        root_dir: path to root directory of this project from wherever this function is called from
        parser_name: optional. Can be used to make the predictions folder if
        predictions_directory: optional. The actual predictions folder. One of the last two must be given.

    Returns:

    """
    pred = load_parser_output(info.subcorpus_filename, root_dir, parser_name=parser_name,
                              predictions_directory=predictions_directory)
    gold = load(f"{root_dir}/corpus/subcorpora/{info.subcorpus_filename}.txt")
    evaluator = eval_class(gold, pred, root_dir, info)
    rows = evaluator.run_evaluation()
    return rows


def print_pretty_table(result_rows):
    table = PrettyTable()
    table.field_names = ["Dataset", "Metric", "Score", "Wilson CI", "Sample size"]
    table.align = "l"
    for row in result_rows:
        if row[0] is None:
            category = ""
        elif isinstance(row[0], str):
            category = row[0]
        else:
            category = row[0].display_name
        eval_type = _get_row_evaluation_type(row)
        if eval_type == EVAL_TYPE_SUCCESS_RATE:
            wilson_ci = wilson_score_interval(row[3], row[4])
            if row[4] > 0:
                table.add_row([category, row[1], num_to_score_with_preceding_0(row[3] / row[4]),
                               f"[{num_to_score_with_preceding_0(wilson_ci[0])}, {num_to_score_with_preceding_0(wilson_ci[1])}]", row[4]])
            else:
                print("Division by zero!", file=sys.stderr)
                print(row[0].display_name, row[1:], file=sys.stderr)
        elif eval_type == EVAL_TYPE_F1:
            if len(row) > 4:
                sample_size = row[4]
            else:
                sample_size = ""
            table.add_row([category, row[1], num_to_score_with_preceding_0(row[3]), "", sample_size])
        elif eval_type == 1:
            table.add_row(["", "", "", "", ""])
            table.add_row([category, "", "", "", ""])
        else:
            print(row)
            raise Exception(f"Unknown evaluation type: {eval_type}")
    print(table)


def make_latex_table(root_dir: str):
    """
    TODO This is not currently working with the refactoring
    Might not matter: I think the csv2latex one works
    """
    result_rows_by_parser_name = pickle.load(open(root_dir + "/results_table.pickle", "rb"))

    master_parser = parser_names[0]
    master_rows = result_rows_by_parser_name[master_parser]

    # results_rows_by_column = zip()
    # print("\n###", list(results_rows_by_column))
    # results_rows_by_column = zip(result_rows_by_parser_name["amparser"],
    #                              # result_rows_by_parser_name["cailam"],
    #                              # result_rows_by_parser_name["amrbart"]
    #                              )

    set_to_scores = dict()
    current_scores = None

    with open(root_dir + "/latex_results_table.txt", "w") as f:
        set_id = ""
        shade_row = True
        for j, parser_row in enumerate(master_rows):
            is_title_row = len(parser_row) == 1
            if is_title_row:
                set_id = parser_row[0][0]
                current_scores = [[] for _ in range(len(parser_names))]
                set_to_scores[parser_row[0]] = current_scores
                continue

            if parser_row[0] is None:
                dataset_name = ""
            elif isinstance(parser_row[0], str):
                dataset_name = parser_row[0]
            else:
                dataset_name = parser_row[0].get_latex_display_name()
            # dataset_name = parser_rows[0][0]
            metric_name = parser_row[1]

            is_unlabeled_edge_row = metric_name == "Unlabeled edge recall"
            is_smatch_row = "smatch" in metric_name.lower()
            if is_unlabeled_edge_row or is_smatch_row:
                continue

            is_sanity_check_row = "sanity" in dataset_name.lower()
            is_prereq_row = "prereq" in metric_name.lower()
            if set_id in ["1", "3", "4", "7"]:  # the sets that start a new table
                shade_row = True
            else:
                if is_sanity_check_row or dataset_name.strip() == "":
                    pass
                else:
                    shade_row = not shade_row

            shading_prefix = "\\rowcolor{lightlightlightgray}" if shade_row else ""
            latex_line = f"\t\t{shading_prefix}{set_id} & {dataset_name} & {metric_name}"

            is_success_rate_row = parser_row[2] == EVAL_TYPE_SUCCESS_RATE
            if is_success_rate_row and not "precision" in metric_name.lower():
                for name in parser_names:
                    assert parser_row[4] == result_rows_by_parser_name[name][4]
            for i in range(len(parser_names)):
                if is_success_rate_row:
                    wilson_ci = wilson_score_interval(row[3], row[4])
                    score = num_to_score_with_preceding_0(row[3] / row[4])
                    if not (is_prereq_row or is_sanity_check_row):
                        current_scores[i].append(row[3] / row[4])
                    if len(score) == 2:
                        score = "\\phantom{1}" + score
                    lower_bound = num_to_score_with_preceding_0(wilson_ci[0])
                    upper_bound = num_to_score_with_preceding_0(wilson_ci[1])
                    if len(lower_bound) == 2:
                        lower_bound = lower_bound
                    if len(upper_bound) == 2:
                        upper_bound = "\\phantom{1}" + upper_bound
                    # latex_line += f" & ${score}_{{{lower_bound}}}^{{{upper_bound}}}$"
                    latex_line += f" & \\successScore{{{score}}}{{{lower_bound}}}{{{upper_bound}}}{{\\phantom{{1}}}}"
                else:
                    if not (is_prereq_row or is_sanity_check_row):
                        current_scores[i].append(row[3])
                    latex_line += f" & {num_to_score_with_preceding_0(row[3])}\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ "

            if is_success_rate_row:
                latex_line += f" & {parser_rows[0][4]}\\\\\n"
            else:
                latex_line += f" & \\\\\n"

            f.write(latex_line)

            set_id = ""

    with open(root_dir + "/latex_compact_table.txt", "w") as f:
        for set_name, scores_by_parser in set_to_scores.items():
            # print(set_name)
            # print(scores_by_parser)
            latex_line = f"\t\t{set_name}"
            for scores in scores_by_parser:
                # average the scores
                score = sum(scores) / len(scores)
                latex_line += f" & {num_to_score_with_preceding_0(score)}"
            latex_line += "\\\\\n"
            f.write(latex_line)



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



def main():
    create_results_pickles()
    # make_latex_table(results_path)

if __name__ == '__main__':
    main()