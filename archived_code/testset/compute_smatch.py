from evaluation.corpus_metrics import compute_smatch_f


def main():
    for parsername in ["amparser", "cailam", "amrbart"]:
        print(parsername)
        print(compute_smatch_f("../../corpus/testset.txt", f"../../{parsername}-output/testset.txt"))


if __name__ == "__main__":
    main()
