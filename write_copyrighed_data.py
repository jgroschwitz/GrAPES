"""
writes decrypted licensed data for unbounded dependencies
"""
import penman


def write_ptb_data(decrypted_txt, decrypted_tsv):
    """
    Write the contents of the AMR and TSV files for descripted PTB sentences to files
    individual AMR corpus:
        corpus/subcorpora/unbounded_dependencies.txt
    append to corpus/corpus.txt
    TSV: corpus/unbounded_dependencies.tsv

    Args:
        decrypted_txt: the content of the AMR file
        decrypted_tsv: the content of the TSV file
    """
    amrs = penman.loads(decrypted_txt)
    subcorpus = "unbounded_dependencies"

    for i, entry in enumerate(amrs):
        # copy old ID to supplementary info
        entry.metadata["suppl"] = entry.metadata["id"]
        entry.metadata["id"] = f"{subcorpus}_{i}"

    # write to corpus files
    with open(f"corpus/subcorpora/{subcorpus}.txt", 'w') as subcorpus_file:
        penman.dump(amrs, subcorpus_file)
    with open(f"corpus/test_corpus.txt", 'a') as corpus_file:
        corpus_file.write("\n")  # need blank line in between
        penman.dump(amrs, corpus_file)

    # write to TSV
    with open(f"corpus/{subcorpus}.tsv", 'w') as tsv_file:
        tsv_file.write(decrypted_tsv)


def update_from_amr_testset(sentences, subcorpus_name):
    with open(f"corpus/subcorpora/{subcorpus_name}.txt", 'r') as subcorpus_file:
        amrs = penman.load(subcorpus_file)
        for g in amrs:
            if g.metadata["snt"] == "(removed -- see documentation)":
                print(g.metadata["suppl"])



if __name__ == "__main__":
    # the strings to write to files
    txt = """# CCG unbounded dependencies, annotated by Meaghan and students, built from relatives_meaghan.tsv object-free-relatives_chris.tsv object-relative-null_chris.tsv object_wh_questions_chris.tsv right_node_raising_chris.tsv subj_relative_embedded_chris.tsv subj_relatives_chris.tsv 

# ::id cf21.mrg-or-1-0
# ::snt We have also developed techniques for recognizing and locating underground nuclear tests through the waves in the ground which they generate .
# ::cat object relative
# ::distance 6
(r / dummy)

# ::id cf03.mrg-or-3-1
# ::snt But these are dreamed in original action , in some particular continuity which we don't remember having seen in real life .
# ::cat object relative
# ::distance 7
(r / dummy)
    """
    tsv = "the\ttsv\nmore\tstuff"

    # write_ptb_data(txt, tsv)

    update_from_amr_testset([], )
