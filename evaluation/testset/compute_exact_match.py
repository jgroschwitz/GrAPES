from penman import load
from evaluation.graph_matcher import equals_modulo_isomorphy


def main():
    golds = load("../../corpus/testset.txt")
    for parsername in ["amparser", "amrbart", "cailam"]:
        print(parsername)
        preds = load(f"../../{parsername}-output/testset.txt")
        exact_match = 0
        total = 0

        many_words_exact_match = 0
        many_words_total = 0

        for pred, gold in zip(preds, golds):
            many_words = len(gold.metadata["snt"].split(" ")) >= 15
            total += 1
            if many_words:
                many_words_total += 1
            if equals_modulo_isomorphy(gold, pred):
                exact_match += 1
                if many_words:
                    many_words_exact_match += 1

        print(f"Exact match: {exact_match}/{total} = {exact_match/total}")
        print(f"Exact match (15+ words): {many_words_exact_match}/{many_words_total} = {many_words_exact_match/many_words_total}")


if __name__ == "__main__":
    main()
