import argparse
from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.i_pragmatic_reentrancies import PragmaticReentrancies
from evaluation.full_evaluation.category_evaluation.ii_unambiguous_reentrancies import UnambiguousReentrancies
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.single_eval import num_to_score

#  Category names are the same as in the paper (tables 3-5), but all lowercase, and with all punctuation, brackets etc.
#  removed. Except for '+', which is replaced by 'plus', and whitespace ' ' which is replaced by '_'.
#  Sanity checks include the name of the category they are checking, such as multiple_adjectives_sanity_check.
category_name_to_set_class_and_eval_function = {
    "pragmatic_coreference_testset": (PragmaticReentrancies, PragmaticReentrancies.compute_testset_results),
    "pragmatic_coreference_winograd": (PragmaticReentrancies, PragmaticReentrancies.compute_winograd_results),
    "syntactic_gap_reentrancies": (UnambiguousReentrancies, UnambiguousReentrancies.compute_syntactic_gap_results),
    "unambiguous_coreference": (UnambiguousReentrancies, UnambiguousReentrancies.compute_unambiguous_coreference_results),
}


def get_formatted_category_names():
    return "\n".join(category_name_to_set_class_and_eval_function.keys())  # TODO linebreak doesn't seem to work in help


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single category.")
    parser.add_argument('-c', '--category_name', type=str, help='Category to evaluate. Possible values are: '
                                                                + get_formatted_category_names())
    parser.add_argument('-g', '--gold_amr_file', type=str, help='Path to gold AMR file')
    parser.add_argument('-p', '--predicted_amr_file', type=str, help='Path to predicted AMR file. Must contain AMRs '
                                                                     'for all sentences in the gold file, in the same '
                                                                     'order.')
    args = parser.parse_args()
    return args


def get_results(gold_graphs, predicted_graphs, category_name):
    set_class, eval_function = category_name_to_set_class_and_eval_function["pragmatic_coreference_winograd"]
    set = set_class(gold_graphs, predicted_graphs, None, "./")
    return eval_function(set, gold_graphs, predicted_graphs)


def main():
    args = parse_args()
    gold_graphs = load(args.gold_amr_file)
    predicted_graphs = load(args.predicted_amr_file)
    if len(gold_graphs) != len(predicted_graphs):
        raise ValueError("Gold and predicted AMR files must contain the same number of AMRs."
                         "Got " + str(len(gold_graphs)) + " gold AMRs and " + str(len(predicted_graphs))
                         + " predicted AMRs.")
    results = get_results(gold_graphs, predicted_graphs, args.category_name)
    print("Results on " + args.category_name)
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
