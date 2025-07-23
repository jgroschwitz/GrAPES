"""
writes decrypted licensed data for unbounded dependencies and word disambiguation
"""
import penman

# use test_corpus for testing
# main_corpus = "test_corpus"
main_corpus = "corpus"
# path_to_corpus_folder = "../../corpus/"  # if using this script directly
path_to_corpus_folder = "corpus/"  # path from import location


def write_ptb_data(decrypted_txt, decrypted_tsv):
    """
    Write the contents of the AMR and TSV files for decrypted PTB sentences to files
    individual AMR corpus:
        corpus/subcorpora/unbounded_dependencies.txt
    append to corpus/corpus.txt
    TSV: corpus/unbounded_dependencies.tsv

    Args:
        decrypted_txt: the content of the AMR file
        decrypted_tsv: the content of the TSV file
    """
    subcorpus = "unbounded_dependencies"

    # check if we already have everything in place
    append = True
    create_txt = False
    create_tsv = False
    existing_main_corpus = penman.load(f"{path_to_corpus_folder}{main_corpus}.txt")
    sample_suppl = "cf21.mrg-or-1-0"
    found = [g for g in existing_main_corpus if g.metadata["suppl"] == sample_suppl]
    if len(found) > 0:
        print("looks like the entries are already in corpus.txt.")
        append = False
    try:
        f = open(f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt", 'r')
        f.close()
        print(f"looks like you've already got the corpus/subcorpora/{subcorpus}.txt file")
    except FileNotFoundError:
        create_txt = True

    try:
        f = open(f"{path_to_corpus_folder}/{subcorpus}.tsv", 'r')
        f.close()
        print(f"looks like you've already got the corpus/{subcorpus}.tsv file")
    except FileNotFoundError:
        create_tsv = True

    if not (append or create_txt or create_tsv):
        print("Nothing to do; exiting")
        return

    amrs = penman.loads(decrypted_txt)
    print(f"made {len(amrs)} AMRs")

    for i, entry in enumerate(amrs):
        # copy old ID to supplementary info
        entry.metadata["suppl"] = entry.metadata["id"]
        entry.metadata["id"] = f"{subcorpus}_{i}"

    # write to corpus files if needed
    if create_txt:
        penman.dump(amrs, f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt")
        print(f"added corpus/subcorpora/{subcorpus}.txt")
    if append:
        with open(f"{path_to_corpus_folder}/{main_corpus}.txt", 'a') as corpus_file:
            corpus_file.write("\n")  # need blank line in between
            penman.dump(amrs, corpus_file)
            print(f"extended corpus/{main_corpus}.txt")

    # write to TSV if needed
    if create_tsv:
        with open(f"{path_to_corpus_folder}/{subcorpus}.tsv", 'w') as tsv_file:
            tsv_file.write(decrypted_tsv)
            print(f"added corpus/{subcorpus}.tsv")


def update_from_amr_testset(path_to_testset):
    """
    Updates the word_disambiguation subcorpus with the sentences from the test set
    Args:
        path_to_testset: path to the AMR 3.0 testset concatenated file
    """
    subcorpus = "word_disambiguation"
    # check if we already have everything in place
    append = True
    create_txt = False
    existing_main_corpus = penman.load(f"{path_to_corpus_folder}/{main_corpus}.txt")
    sample_suppl = "PROXY_NYT_ENG_20081128_0005.6"
    found = [g for g in existing_main_corpus if g.metadata["suppl"] == sample_suppl]
    if len(found) > 0:
        print("looks like the entries are already in corpus.txt.")
        append = False
    try:
        f = open(f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt", 'r')
        f.close()
        print(f"looks like you've already got the corpus/subcorpora/{subcorpus}.txt file")
    except FileNotFoundError:
        create_txt = True

    if not (create_txt or append):
        print("Nothing to do; exiting")
        return

    test_set = penman.load(path_to_testset)
    with open(f"{path_to_corpus_folder}/{subcorpus}_clean.txt", 'r') as subcorpus_file:
        amrs = penman.load(subcorpus_file)
        for g in amrs:
            if g.metadata["snt"] == "(removed -- see documentation)":
                id = g.metadata["suppl"]
                sentence = [gr.metadata["snt"] for gr in test_set if gr.metadata["id"] == id][0]
                g.metadata["snt"] = sentence

    if create_txt:
        penman.dump(amrs, f"{path_to_corpus_folder}/subcorpora/{subcorpus}.txt")
        print(f"added corpus/subcorpora/{subcorpus}.txt")
    if append:
        with open(f"{path_to_corpus_folder}/{main_corpus}.txt", "a") as c:
            c.write("\n")  # need blank line in between
            penman.dump(amrs, c)
            print(f"extended corpus/{main_corpus}.txt")


