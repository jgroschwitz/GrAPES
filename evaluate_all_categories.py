import argparse
from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.i_pragmatic_reentrancies import PragmaticReentrancies
from evaluation.full_evaluation.category_evaluation.ii_unambiguous_reentrancies import UnambiguousReentrancies
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.single_eval import num_to_score

from evaluate_single_category import category_name_to_set_class_and_eval_function

set_names_with_category_names = [("1. Pragmatic reentrancies", ["pragmatic_coreference_testset", "pragmatic_coreference_winograd"]),
                                    ("2. Unambiguous reentrancies", ["syntactic_gap_reentrancies", "unambiguous_coreference"])
]

category_names_to_source_corpus_name = {
    "pragmatic_coreference_testset": "testset",
    "pragmatic_coreference_winograd": "grapes",
    "syntactic_gap_reentrancies": "testset",
    "unambiguous_coreference": "testset"
}

def get_formatted_category_names():
    return "\n".join(category_name_to_set_class_and_eval_function.keys())  # TODO linebreak doesn't seem to work in help



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all categories.")
    parser.add_argument('-gt', '--gold_amr_testset_file', type=str, help='Path to gold AMR file (testset). A single '
                                                                         'file containing all AMRs of the AMRBank 3.0'
                                                                         'testset.')
    parser.add_argument('-gg', '--gold_amr_grapes_file', type=str, help='Path to GrAPES gold corpus file')
    parser.add_argument('-pt', '--predicted_amr_testset_file', type=str, help='Path to predicted AMR file. Must contain AMRs '
                                                                     'for all sentences in the gold file, in the same '
                                                                     'order.')
    parser.add_argument('-pg', '--predicted_amr_grapes_file', type=str, help='Path to predicted AMR file. Must contain AMRs '
                                                                     'for all sentences in the gold grapes corpus file, in the same '
                                                                     'order.')
    args = parser.parse_args()
    return args


def get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes):
    set_name = "Pragmatic coreference"



def main():
    args = parse_args()
    gold_graphs_testset = load(args.gold_amr_testset_file)
    gold_graphs_grapes = load(args.gold_amr_grapes_file)
    predicted_graphs_testset = load(args.predicted_amr_testset_file)
    predicted_graphs_grapes = load(args.predicted_amr_grapes_file)
    if len(gold_graphs_testset) != len(predicted_graphs_testset):
        raise ValueError("Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the testset here."
                         "Got " + str(len(gold_graphs_testset)) + " gold AMRs and " + str(len(predicted_graphs_testset))
                         + " predicted AMRs.")
    if len(gold_graphs_grapes) != len(predicted_graphs_grapes):
        raise ValueError("Gold and predicted AMR files must contain the same number of AMRs. This is not the case for the grapes corpus here."
                         "Got " + str(len(gold_graphs_grapes)) + " gold AMRs and " + str(len(predicted_graphs_grapes))
                         + " predicted AMRs.")

    results = get_results(gold_graphs_testset, gold_graphs_grapes, predicted_graphs_testset, predicted_graphs_grapes)
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
