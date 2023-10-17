from penman import load


def main():
    """
    Reads the handwritten file corpus/unseen_senses_new_sentences.txt and writes the corresponding TSV file with
    the relevant senses automatically extracted.
    :return:
    """
    graphs = load("../../corpus/unseen_senses_new_sentences.txt")
    with open("../../corpus/unseen_senses_new_sentences.tsv", 'w') as tsv_file:
        for graph in graphs:
            tsv_file.write(f"{graph.metadata['id']}\t{graph.metadata['sense']}\n")


if __name__ == "__main__":
    main()
