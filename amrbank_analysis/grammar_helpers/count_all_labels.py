from collections import Counter

from amrbank_analysis.util import load_corpus_from_folder, get_node_by_name, get_name


def main():
    training_corpus = load_corpus_from_folder("../../../../data/Edinburgh/amr3.0/data/amrs/split/training/")
    print(len(training_corpus))

    label_counter = Counter()
    for graph in training_corpus:
        for instance in graph.instances():
            label_counter[instance.target] += 1

    with open("outputs/all_labels.txt", "w") as f:
        for label, count in label_counter.most_common():
            f.write(f"{count}\t{label}\n")


if __name__ == '__main__':
    main()
