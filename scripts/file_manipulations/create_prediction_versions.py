import penman

minimal_corpus_length = 1471
# added in this order by Jonas's script
unbounded_dependencies_length = 66
word_disambiguation_length = 47

full_corpus_length = 1584

# sanity check
print(full_corpus_length - word_disambiguation_length - unbounded_dependencies_length == minimal_corpus_length)

full_test_corpus = penman.load("../../corpus/test_corpus.txt")
print(len(full_test_corpus))

minimal_corpus = penman.load("../../corpus/corpus.txt")
print(len(minimal_corpus))

full_am_corpus = penman.load("../../../amr-challenge/amparser-output/full_corpus.txt")

print(len(full_am_corpus))

c = penman.load("../../corpus/subcorpora/unbounded_dependencies.txt")

print(len(c))

am_no_word_disambig = full_am_corpus[:-word_disambiguation_length]
print(len(am_no_word_disambig) + word_disambiguation_length)
gold_no_word_disambig = full_test_corpus[:-word_disambiguation_length]

am_no_unbounded = full_am_corpus[:len(minimal_corpus)] + full_test_corpus[len(am_no_word_disambig):]
print(len(am_no_unbounded) + unbounded_dependencies_length)
gold_no_unbounded = full_test_corpus[:len(minimal_corpus)] + full_test_corpus[len(am_no_word_disambig):]

am_minimal_corpus = full_am_corpus[:-(word_disambiguation_length + unbounded_dependencies_length)]
print(len(am_minimal_corpus))

penman.dump(gold_no_unbounded, "../../corpus/no_unbounded.txt")
penman.dump(gold_no_word_disambig, "../../corpus/no_word_disambig.txt")

#penman.dump(am_no_unbounded, "../../../amr-challenge/amparser-output/no_unbounded.txt")
#penman.dump(am_no_word_disambig, "../../../amr-challenge/amparser-output/no_word_disambig.txt")
#penman.dump(am_minimal_corpus, "../../../amr-challenge/amparser-output/minimal.txt")

