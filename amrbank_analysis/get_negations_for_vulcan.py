from util import turn_attributes_into_nodes, load_corpus_from_folder
from vulcan_pickle_builder import VulcanPickleBuilder


def main():
    test_corpus = load_corpus_from_folder("../../../data/Edinburgh/amr3.0/data/amrs/split/test/")
    print(len(test_corpus))

    negation_count = 0

    pickle_builder = VulcanPickleBuilder()

    for graph in test_corpus:
        turn_attributes_into_nodes(graph)
        negation_nodes = []
        for instance in graph.instances():
            if instance.target == "-" and any(
                    e.role == ":polarity" and e.target == instance.source for e in graph.edges()):
                negation_count += 1
                negation_nodes.append(instance.source)

        found_negation = len(negation_nodes) > 0
        if found_negation:
            pickle_builder.add_graph(graph, add_sent=False)
            pickle_builder.add_graph_highlight(negation_nodes)
            split_sentence = graph.metadata["snt"].split(" ")
            negation_word_ids = [i for i, token in enumerate(split_sentence) if is_negation_word(token)]
            pickle_builder.add_sent(split_sentence)
            pickle_builder.add_sent_highlight(negation_word_ids)

    print(f"Found a total of {negation_count} negations")
    pickle_builder.save_pickle("outputs/negations_testset.pkl")


def is_negation_word(word):
    negation_words = {"no", "non", "not", "dont", "without", "aint", "couldnt", "cant", "wont", "wouldnt", "doesnt",
                      "didnt", "isnt", "wasnt", "werent", "hasnt", "hadnt", "shouldnt", "neither", "nor", "none",
                      "never", "nobody", "nothing", "nowhere", "noone"}
    return word.lower() in negation_words or word.lower().endswith("n't")


if __name__ == "__main__":
    main()
