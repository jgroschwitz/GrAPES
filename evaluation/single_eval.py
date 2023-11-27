import sys

from penman import load
from evaluation.pp_attachment import evaluate_pp_attachments
from evaluation.long_lists import evaluate_long_lists, evaluate_singletons, evaluate_long_lists_generalization
import prettytable
from evaluation.corpus_metrics import compute_exact_match_fraction, compute_smatch_f


def main(args):
    parser = args[1]
    dataset = args[2]

    gold_file_path = f"../corpus/{dataset}.txt"
    golds = load(gold_file_path)

    prediction_folder = "../" + parser + "-output/"
    prediction_file_path = prediction_folder + dataset + ".txt"
    predictions = load(prediction_file_path)

    print(len(golds))
    print(len(predictions))

    if dataset == "pp_attachment":
        results = evaluate_pp_attachments("../corpus/", prediction_folder)
        print_pp_attachment_results(parser, results)
    elif dataset == "long_lists":
        results = evaluate_long_lists(predictions, golds)
        print_list_results(parser, results)
        results_generalization = evaluate_long_lists_generalization(predictions, golds)
        print("Generalization:")
        print_list_results(parser, results_generalization)
    elif dataset == "long_lists_singletons":
        evaluate_singletons(predictions, golds)  # prints results
    elif dataset in ["centre_embedding", "adjectives", "adjectives_sanity_check", "nested_control",
                     "nested_control_small", "long_lists_sanity_check"]:
        print("Exact match, modulo edge labels and propbank senses (on a scale of 0-1):")
        print(compute_exact_match_fraction(golds, predictions, match_edge_labels=False, match_senses=False))
        print("Smatch score:")
        print(compute_smatch_f(gold_file_path, prediction_file_path))
    elif dataset == "testset" or dataset == "testsetAMR2_0":
        print("Smatch score:")
        print(compute_smatch_f(gold_file_path, prediction_file_path))
    else:
        print(f"Dataset {dataset} not recognized. Exiting single_eval.py")


def print_list_results(parser, results):
    results_table = prettytable.PrettyTable()
    results_table.add_column("Long Lists", [":opi F1",
                                            "Conjunct F1"])
    results_table.align = "l"
    results_table.add_column(parser,
                             [num_to_score(results[0][2]),
                              num_to_score(results[1][2])])
    print(results_table)


def print_pp_attachment_results(parser, results):
    results_table = prettytable.PrettyTable()
    results_table.add_column("PP-attachment", ["Requirements",
                                               "UAS",
                                               "LAS"])
    results_table.align = "l"
    results_table.add_column(parser,
                             [num_to_score(results[0]),
                              num_to_score(results[1]),
                              num_to_score(results[2])])
    print(results_table)


def num_to_score(number):
    return f"{(number * 100):.0f}"


if __name__ == "__main__":
    main(sys.argv)
