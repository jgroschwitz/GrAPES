import argparse
import os
import sys

from penman import load

from evaluate_all_categories import pretty_print_structural_generalisation_by_size
from evaluation.category_metadata import category_name_to_set_class_and_metadata
from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.run_full_evaluation import evaluate
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.single_eval import num_to_score
from evaluation.structural_generalization import size_mappers


def get_formatted_category_names():
    return "\n".join(category_name_to_set_class_and_metadata.keys())  # TODO linebreak doesn't seem to work in help


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single category.")
    parser.add_argument('-c', '--category_name', type=str, help='Category to evaluate. Possible values are: '
                                                                + get_formatted_category_names())
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

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    eval_class, info = category_name_to_set_class_and_metadata[args.category_name]
    predictions_path = args.predicted_amr_file

    # if args.predicted_amr_file_or_directory is not None:
    #     if os.path.isdir(args.predicted_amr_file_or_directory):
    #         print("Using predictions directory", args.predicted_amr_file_or_directory)
    #         predicted_path = args.predicted_amr_file_or_directory
    #         predicted_testset_path = None
    #     elif os.path.isfile(args.predicted_amr_file_or_directory):
    #         print("Using predicted file", args.predicted_amr_file_or_directory)
    #         predicted_testset_path = args.predicted_amr_file_or_directory
    #         predicted_path = None
    #     else:
    #         print("Predicted AMR file or directory not found.")
    #         exit(1)

    if predictions_path.endswith(f"{info.subcorpus_filename}.txt"):
        print("Using predicted AMR subcorpus file", predictions_path)
    else:
        print("Presumably this is the full GrAPES or AMR 3.0 testest parser output file: ", predictions_path)

    if args.gold_amr_file is not None:
        gold_amrs = load(args.gold_amr_file)
    else:
        gold_amrs = load("corpus/corpus.txt")

    predicted_amrs = load(predictions_path)
    predictions_directory = os.path.dirname(predictions_path)

    print(f"Results on {info.display_name}:\n")
    evaluator = eval_class(gold_amrs, predicted_amrs, ".", info)
    print(info.subcorpus_filename, evaluator.__class__.__name__)

    # Structural generalisation results by size
    if info.subtype == "structural_generalization" and info.subcorpus_filename in size_mappers:
        print(info.subcorpus_filename)
        generalisation_by_size = evaluator.get_results_by_size()
        pretty_print_structural_generalisation_by_size({info.subcorpus_filename: generalisation_by_size})

    results = evaluate(evaluator, info, root_dir=".", predictions_directory=predictions_directory)

    for row in results:
        metric_name = row[1]
        metric_type = row[2]
        if metric_type == EVAL_TYPE_SUCCESS_RATE:
            wilson_ci = wilson_score_interval(row[3], row[4])
            if row[4] > 0:
                print(f"{metric_name}: {num_to_score(row[3] / row[4])} with Wilson confidence interval "
                      f"[{num_to_score(wilson_ci[0])}, {num_to_score(wilson_ci[1])}] and sample size {row[4]}])")
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


if __name__ == "__main__":
    main()
