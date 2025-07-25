import argparse
import os

from penman import load

from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
     is_testset_category, get_formatted_category_names_by_main_file, \
    add_sanity_check_suffix
from evaluation.full_evaluation.category_evaluation.subcategory_info import is_sanity_check
from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE, \
    size_mappers, STRUC_GEN, EVAL_TYPE_PRECISION
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo
from evaluation.full_evaluation.run_full_evaluation import evaluate, structural_generalisation_by_size_as_table, \
    load_predictions
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.util import num_to_score, SANITY_CHECK
from scripts.argparse_formatter import SmartFormatter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single category. Make sure you provide paths to the AMR 3.0 testset for AMR 3.0 testset categories and GrAPES for GrAPES ", formatter_class=SmartFormatter,
                                     )
    parser.add_argument('-c', '--category_name', type=str, help='Category to evaluate. Possible values are:\n'
                                                                + get_formatted_category_names_by_main_file())
    parser.add_argument('-g', '--gold_amr_file', type=str, help='Path to gold AMR file. '
                                                                'Optional if a GrAPES-specific category, '
                                                                'in which case we use corpus/corpus.txt',
                        default=None)
    parser.add_argument('-p', '--predicted_amr_file',
                        type=str,
                        help="Path to the predicted AMR file. May use the full corpus output (GrAPES or AMR 3.0)"
                             " or a GrAPES subcorpus file")
    parser.add_argument('-n', '--parser_name', type=str,
                        help="name of parser (optional)", default="parser")
    parser.add_argument("-e", "--error_analysis", action="store_true", help="Pickle correct and incorrect graph ids")
    parser.add_argument("-s", "--smatch", action="store_true", help="Calculate Smatch even if we normally wouldn't for this category ")

    args = parser.parse_args()
    return args

def instance_info_from_args(args):
    instance_instructions = EvaluationInstanceInfo(
        absolute_path_to_predictions_file=args.predicted_amr_file,
        absolute_path_to_gold_file=args.gold_amr_file,
        do_error_analysis=args.error_analysis,
        parser_name=args.parser_name,
        run_smatch=args.smatch,
        run_structural_generalisation_smatch=args.smatch
    )
    return instance_instructions

def main():
    args = parse_args()
    instance_info = instance_info_from_args(args)

    eval_class, info = category_name_to_set_class_and_metadata[args.category_name]


    gold_path, predictions_path, use_subcorpus = get_gold_path_based_on_info(args.gold_amr_file, info, instance_info)

    gold_amrs = load(gold_path)
    predicted_amrs = load_predictions(predictions_path)

    evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)
    results = evaluate(evaluator, info, instance_info)
    assert len(results) > 0, "No results!"

    caption = f"\nResults on {info.display_name}"

    # Structural generalisation results by size
    if info.subtype == STRUC_GEN:
        do_by_size = info.subcorpus_filename in size_mappers
        if do_by_size:
            generalisation_by_size = evaluator.get_results_by_size()
            structural_generalisation_by_size_as_table({info.subcorpus_filename: generalisation_by_size})

        if not is_sanity_check(info):
            # Try doing the sanity check for a main class
            try:
                eval_class, info = category_name_to_set_class_and_metadata[add_sanity_check_suffix(args.category_name)]
                if use_subcorpus:
                    gold_amrs = load(f"corpus/subcorpora/{info.subcorpus_filename}.txt")
                    predicted_amrs = load(f"{instance_info.predictions_directory_path()}/{info.subcorpus_filename}.txt")
                evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)
                new_rows = evaluate(evaluator, info, instance_info)
                results += new_rows
                caption += " and Sanity Check"
            except Exception as e:
                print("(No Sanity Check: Need full GrAPES corpus or separate sanity_check file)")
                raise e

    print(caption)
    print()

    for row in results:
        info = row[0]
        metric_name = row[1]
        metric_type = row[2]
        if info is not None and info.display_name == SANITY_CHECK:
            metric_name = f"{SANITY_CHECK} {metric_name}"
        if metric_type in [EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_PRECISION]:
            wilson_ci = wilson_score_interval(row[3], row[4])
            total_type = "sample size" if metric_type == EVAL_TYPE_SUCCESS_RATE else "total predictions"
            if row[4] > 0:
                print(f"{metric_name}: {num_to_score(row[3] / row[4])} with Wilson confidence interval "
                      f"[{num_to_score(wilson_ci[0])}, {num_to_score(wilson_ci[1])}] and {total_type} {row[4]}")
            else:
                print("ERROR: Division by zero! This means something unexpected went wrong (feel free to contact the "
                      "developers of GrAPES for help, e.g. by filing an issue on GitHub).")
                print(row)
        elif metric_type == EVAL_TYPE_F1:
            print(f"{metric_name}: {num_to_score(row[3])}")
        else:
            print("ERROR: Unexpected evaluation type! This means something unexpected went wrong (feel free to "
                  "contact the developers of GrAPES for help, e.g. by filing an issue on GitHub).")
            print(row)


def get_gold_path_based_on_info(given_gold_path, info, instance_info):
    predictions_path = instance_info.pred_grapes_file_path()
    prediction_file_name = os.path.basename(predictions_path)[:-4]
    if info.filename_belongs_to_subcategory(prediction_file_name):
        use_subcorpus = True
        instance_info.given_single_file = True
    else:
        use_subcorpus = False
    if is_testset_category(info):
        if instance_info.gold_testset_path() is None:
            print(f"No gold AMR 3.0 testset file provided for testset category {info.name}; exiting")
            exit(1)
    if given_gold_path is not None:
        gold_path = given_gold_path
    elif use_subcorpus:
        print("using gold subcorpus", prediction_file_name)
        gold_path = f"{instance_info.root_dir}/corpus/subcorpora/{prediction_file_name}.txt"
    else:
        gold_path = instance_info.gold_grapes_path()
    return gold_path, predictions_path, use_subcorpus


if __name__ == "__main__":
    main()