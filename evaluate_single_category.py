import argparse
import os

from penman import load

from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, get_formatted_category_names
from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.run_full_evaluation import evaluate, pretty_print_structural_generalisation_by_size, \
    load_parser_output
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.util import num_to_score
from evaluation.novel_corpus.structural_generalization import size_mappers, add_sanity_check_suffix


class SmartFormatter(argparse.HelpFormatter):
    """
    Custom Help Formatter used to split help text when '\n' was
    inserted in it.
    """
    def _split_lines(self, text, width):
        r = []
        for t in text.splitlines(): r.extend(argparse.HelpFormatter._split_lines(self, t, width))
        return r


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single category.", formatter_class=SmartFormatter)
    parser.add_argument('-c', '--category_name', type=str, help='Category to evaluate. Possible values are:\n'
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

    if predictions_path.endswith(f"{info.subcorpus_filename}.txt"):
        print("Using predicted AMR subcorpus file", predictions_path)
    else:
        print("Presumably this is the full GrAPES or AMR 3.0 testest parser output file: ", predictions_path)

    if args.gold_amr_file is not None:
        gold_amrs = load(args.gold_amr_file)
    else:
        gold_amrs = load("corpus/corpus.txt")

    predicted_amrs = load_predictions(predictions_path)
    predictions_directory = os.path.dirname(predictions_path)

    try:
        evaluator = eval_class(gold_amrs, predicted_amrs, ".", info)
    except Exception as e:
        if args.category_name == "cp_recursion_plus_coreference":
            predicted_amrs += load_parser_output("deep_recursion_3s", ".", predictions_directory=predictions_directory)
            new_golds = load("corpus/subcorpora/deep_recursion_3s.txt")
            if len(new_golds) == 0:
                print("No graphs found!")
            gold_amrs += new_golds
            evaluator = eval_class(gold_amrs, predicted_amrs, ".", info)
        else:
            raise e
    results = evaluate(evaluator, info, root_dir=".", predictions_directory=predictions_directory)

    caption = f"\nResults on {info.display_name}"

    # Structural generalisation results by size
    if info.subtype == "structural_generalization":
        do_by_size = info.subcorpus_filename in size_mappers
        if do_by_size:
            generalisation_by_size = evaluator.get_results_by_size()
            pretty_print_structural_generalisation_by_size({info.subcorpus_filename: generalisation_by_size})

        if not args.category_name.endswith("sanity_check"):
            # Try doing the sanity check for a main class
            try:
                eval_class, info = category_name_to_set_class_and_metadata[add_sanity_check_suffix(args.category_name)]
                evaluator = eval_class(gold_amrs, predicted_amrs, ".", info)
                results += evaluate(evaluator, info, root_dir=".", predictions_directory=predictions_directory)
                caption += " and Sanity Check"
            except Exception as e:
                print("(No Sanity Check: Need full or separate file)")

    print(caption)
    print()

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


def load_predictions(predictions_path, encoding="utf8"):
    """
    Add some printing around loading predictions in case of warnings from Penman
    """
    print("\nLoading predicted AMRs...")
    predicted_amrs = load(predictions_path, encoding=encoding)
    print("Done\n")
    return predicted_amrs


if __name__ == "__main__":
    main()