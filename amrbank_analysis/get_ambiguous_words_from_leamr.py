import os
from collections import Counter

from util import load_corpus_with_alignments, LEAMR_ALIGNMENTS, LEAMR_LDC_TRAIN_AMRS, LEAMR_LDC_TEST_AMRS, \
    amr_utils_graph_to_penman_graph_with_all_explicit_names
from vulcan_pickle_builder import VulcanPickleBuilder

from nltk.stem import WordNetLemmatizer
import nltk


def main():
    nltk.download('omw-1.4')
    train_amrs, train_alignments = load_corpus_with_alignments(LEAMR_LDC_TRAIN_AMRS, LEAMR_ALIGNMENTS)
    lemmatizer = WordNetLemmatizer()
    test_amrs, test_alignments = load_corpus_with_alignments(LEAMR_LDC_TEST_AMRS, LEAMR_ALIGNMENTS)

    lemmas2labels_counters_train = get_lemmas2labels_counters(lemmatizer, train_alignments, train_amrs)
    lemmas2labels_counters_test = get_lemmas2labels_counters(lemmatizer, test_alignments, test_amrs)

    most_ambiguous_lemmas = sorted(lemmas2labels_counters_train.keys(),
                                   key=lambda x: sum([len(inner_list) for inner_list in lemmas2labels_counters_train[x].values()])
                                                 - max([len(inner_list) for inner_list in lemmas2labels_counters_train[x].values()]),
                                   reverse=True)

    # create directory outputs/ambiguous_lemmas_vulcan
    os.makedirs("outputs/ambiguous_lemmas_vulcan", exist_ok=True)
    with open("outputs/ambiguous_lemmas.txt", "w") as f:
        for lemma in most_ambiguous_lemmas[:100]:
            vulcan_pickle_builder = VulcanPickleBuilder()
            f.write(lemma + "\n")
            sorted_labels = sorted(lemmas2labels_counters_train[lemma].keys(),
                                   key=lambda x: len(lemmas2labels_counters_train[lemma][x]),
                                   reverse=True)
            max_length = min(len(sorted_labels), 8)
            for label in sorted_labels[:max_length]:
                train_count = len(lemmas2labels_counters_train[lemma][label])
                test_count = len(lemmas2labels_counters_test[lemma][label]) if lemma in lemmas2labels_counters_test \
                                                                and label in lemmas2labels_counters_test[lemma] else 0
                if test_count >= 1:
                    f.write(f"\t{label}\t\t{train_count} ({test_count})\n")
                    for amr, al in lemmas2labels_counters_test[lemma][label]:
                        amr_penman_graph, new_nns = amr_utils_graph_to_penman_graph_with_all_explicit_names(amr)
                        vulcan_pickle_builder.add_graph(amr_penman_graph)
                        vulcan_pickle_builder.add_graph_highlight([new_nns[nn] for nn in al.nodes])
                        vulcan_pickle_builder.add_sent_highlight(al.tokens)
            # if max_length < len(sorted_labels):
            #     f.write(f"\t...({sum([lemmas2labels_counters_train[lemma][l] for l in sorted_labels[8:]])})\n")
            f.write("\n\n")
            vulcan_pickle_builder.save_pickle(f"outputs/ambiguous_lemmas_vulcan/{lemma}.pickle")


def get_lemmas2labels_counters(lemmatizer, train_alignments, train_amrs):
    lemmas2labels_counters = dict()
    for amr in train_amrs:
        alignments = train_alignments[amr.id]
        for al in alignments:
            lemmas_span = " ".join([lemmatizer.lemmatize(amr.tokens[i].lower()) for i in al.tokens])
            label2amrs_here = lemmas2labels_counters.setdefault(lemmas_span, dict())
            # print(al.nodes)
            node_labels = " ".join(sorted([amr.nodes[nn] for nn in al.nodes if nn in amr.nodes]))
            amrs_here = label2amrs_here.setdefault(node_labels, list())
            amrs_here.append((amr, al))
    return lemmas2labels_counters


if __name__ == "__main__":
    main()
