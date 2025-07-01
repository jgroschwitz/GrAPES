from collections import Counter
from typing import List, Callable

from penman import Graph, load

from evaluation.corpus_metrics import compute_exact_match_fraction, compute_smatch_f, \
    compute_exact_match_successes_and_sample_size
from evaluation.graph_matcher import equals_modulo_isomorphy

structural_generalization_corpus_names = ["adjectives", "centre_embedding", "nested_control", "deep_recursion_basic",
                "deep_recursion_pronouns", "deep_recursion_3s", "deep_recursion_rc",
                "deep_recursion_rc_contrastive_coref"]

size_mappers = {"adjectives": lambda x: x - 2,
                "centre_embedding": lambda x: (x - 2) // 2,
                "nested_control": lambda x: x,
                "deep_recursion_basic": lambda x: x - 1,
                "deep_recursion_pronouns": lambda x: x - 1,
                "deep_recursion_3s": lambda x: x - 1,
                "deep_recursion_rc": lambda x: x + 1,
                "deep_recursion_rc_contrastive_coref": lambda x: x + 1}



def add_sanity_check_suffix(filename):
    return f"{filename}_sanity_check"


def get_exact_match_by_size(gold_graphs: List[Graph], predicted_graphs: List[Graph],
                            size_mapper: Callable[[int], int] = lambda x: x):
    assert len(gold_graphs) == len(predicted_graphs)
    correct_counts = Counter()
    total_counts = Counter()
    for gold, prediction in zip(gold_graphs, predicted_graphs):
        size = size_mapper(int(gold.metadata["size0"]))
        total_counts[size] += 1
        if equals_modulo_isomorphy(gold, prediction, match_edge_labels=False, match_senses=False):
            correct_counts[size] += 1
    ret = {size: correct_counts[size] / total_counts[size] for size in sorted(total_counts.keys())}
    ret["total"] = sum(correct_counts.values()) / sum(total_counts.values())
    return ret


def get_all_success_counts(parser_name: str, root_dir="../"):
    """

    :param parser_name:
    :return: Table where the columns are the corpora and the rows contain the metrics. In each row, the first entry
    is the name of the corpus, the second entry is the number of exact matches, the third is sample size and
    the fourth smatch score.
    """
    ret = {}
    for corpus_name in structural_generalization_corpus_names:
        gold_graphs_path = f"{root_dir}/corpus/{corpus_name}.txt"
        predicted_graphs_path = f"{root_dir}/{parser_name}-output/{corpus_name}.txt"
        gold_graphs = load(gold_graphs_path)
        predicted_graphs = load(predicted_graphs_path)
        gold_graphs_sanity_check_path = f"{root_dir}/corpus/{add_sanity_check_suffix(corpus_name)}.txt"
        predicted_graphs_sanity_check_path = f"{root_dir}/{parser_name}-output/{add_sanity_check_suffix(corpus_name)}.txt"
        gold_graphs_sanity_check = load(gold_graphs_sanity_check_path)
        predicted_graphs_sanity_check = load(predicted_graphs_sanity_check_path)

        successes, sample_size = compute_exact_match_successes_and_sample_size(gold_graphs, predicted_graphs,
                                                                               match_edge_labels=False,
                                                                               match_senses=False)

        smatch_f1 = compute_smatch_f(gold_graphs_path, predicted_graphs_path)

        ret[corpus_name] = [successes, sample_size, smatch_f1]

        successes_sc, sample_size_sc = compute_exact_match_successes_and_sample_size(gold_graphs_sanity_check,
                                                                                     predicted_graphs_sanity_check,
                                                                                     match_edge_labels=False,
                                                                                     match_senses=False)

        smatch_f1_sc = compute_smatch_f(gold_graphs_sanity_check_path, predicted_graphs_sanity_check_path)

        ret[corpus_name + "_sanity_check"] = [successes_sc, sample_size_sc, smatch_f1_sc]
    return ret




def main():
    parser_names = ["amparser", "cailam", "amrbart"]

    for parser_name in parser_names:
        print(parser_name)
        for corpus_name in structural_generalization_corpus_names:
            print(corpus_name)

            gold_graphs_path = f"../corpus/{corpus_name}.txt"
            predicted_graphs_path = f"../{parser_name}-output/{corpus_name}.txt"
            gold_graphs = load(gold_graphs_path)
            predicted_graphs = load(predicted_graphs_path)
            gold_graphs_sanity_check_path = f"../corpus/{corpus_name}_sanity_check.txt"
            predicted_graphs_sanity_check_path = f"../{parser_name}-output/{corpus_name}_sanity_check.txt"
            gold_graphs_sanity_check = load(gold_graphs_sanity_check_path)
            predicted_graphs_sanity_check = load(predicted_graphs_sanity_check_path)

            # print("Exact match by size:")
            # print(get_exact_match_by_size(gold_graphs, predicted_graphs, size_mapper=size_mappers[corpus_name]))

            # print("Smatch:")
            # print(compute_smatch_f(gold_graphs_path, predicted_graphs_path))
            # print()

            # print("Sanity Check:")
            # print(compute_exact_match_fraction(gold_graphs_sanity_check, predicted_graphs_sanity_check,
            #                                    match_edge_labels=False, match_senses=False))
            #
            # print("Smatch sanity Check:")
            # print(compute_smatch_f(gold_graphs_sanity_check_path, predicted_graphs_sanity_check_path))
            # print()


if __name__ == "__main__":
    main()
