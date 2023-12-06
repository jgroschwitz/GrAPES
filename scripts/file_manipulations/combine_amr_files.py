"""
Reads in all txt files in the corpus folder and adds them to corpus.txt
with updated IDs
Updates TSV files as necessary with new IDs

Also updates AM parser outputs if needed
"""

import os
import sys

import penman

path_to_existing_amr_files = "../../corpus/subcorpora"
outpath = "../../corpus"
full_corpus = []  # we'll eventually penman.dump this

# if we're also updating AM parser output IDs (run before we updated some IDs)
am = False
am_path = "../amr-challenge/amparser-output"
am_full_corpus = []

# These will end up with different IDs in them
for f in os.listdir(outpath):
    if f in ["unseen_senses_new_sentences.tsv",
             "unseen_roles_new_sentences.tsv",
             "winograd.tsv", "unbounded_dependencies.tsv"]:
        subcorpus = f.split(".")[0]
        os.rename(f"{outpath}/{f}", f"{outpath}/{subcorpus}_old_ids.tsv")

# loop through existing corpus files
for corpus_file in os.listdir(path_to_existing_amr_files):
    if corpus_file.endswith(".txt") and corpus_file not in ["corpus.txt", "word_disambiguation_clean.txt"]:  # corpus_file == "winograd.txt":
        # print("file", corpus_file, file=sys.stderr)
        category = penman.load(f"{path_to_existing_amr_files}/{corpus_file}")
        changed = False  # tracking whether we changed any IDs
        category_name = corpus_file[:-4]
        # print(category_name)
        if corpus_file == "word_disambiguation.txt":
            # there are some test set items in here
            # that we need to remove for licensing reasons
            word_disambiguation = []
            for i, entry in enumerate(category):
                # copy old ID to supplementary info
                entry.metadata["suppl"] = entry.metadata["id"]
                entry.metadata["id"] = f"{category_name}_{i}"
                if not entry.metadata["suppl"].startswith("word_disambiguation"):
                    # this is from the test set
                    entry.metadata["snt"] = "(removed -- see documentation)"
                word_disambiguation.append(entry)
            # write new corpus to word_disambiguation_clean.txt
            penman.dump(word_disambiguation, f"{path_to_existing_amr_files}/word_disambiguation_clean.txt")
        else:
            for i, entry in enumerate(category):
                # copy old ID to supplementary info
                entry.metadata["suppl"] = entry.metadata["id"]
                entry.metadata["id"] = f"{category_name}_{i}"
                if entry.metadata["suppl"] != entry.metadata["id"]:
                    changed = True
            # troubleshooting
                # print(entry.metadata["suppl"], file=sys.stderr)
                # penman.layout.configure(entry)  # try to read it in
            # add this category to the full corpus
            full_corpus += category
            # output to new file so we have the new IDs
            penman.dump(category, f"{outpath}/subcorpora/{corpus_file}")
            if changed:
                print("ids changed in", corpus_file)
            if am:
                # update AM parser output IDs
                if len(category) == 0:
                    # if this wasn't an AMR file, we just get an empty list when we read it in
                    print("*** not an AMR file:\n", corpus_file, "\n")
                else:
                    try:
                        am_predictions = penman.load(f"{am_path}/{corpus_file}")
                        am_full_corpus += am_predictions
                        if len(am_predictions) != len(category):
                            print(f"** {category_name} has {len(am_predictions)} in AM vs {len(category)} in Gold")
                    except FileNotFoundError:
                        print("*** not in AM outputs:\n", corpus_file, "\n")

        # hard coded -- these ones actually need to updated because of new IDs
        if corpus_file in ["unseen_senses_new_sentences.txt",
                           "unseen_roles_new_sentences.txt",
                           "winograd.txt", "unbounded_dependencies.txt"]:
            # update the tsv file with new IDs
            with open(f"{outpath}/{category_name}_old_ids.tsv", 'r') as tsv:
                lines = tsv.readlines()
                new_lines = []
                for line in lines:
                    columns = line.split("\t")
                    id = columns[0]
                    # find the entry
                    for entry in category:
                        if entry.metadata["suppl"] == id:
                            rest_of_line = "\t".join(columns[1:])
                            # replace the ID
                            line = f"{entry.metadata['id']}\t{rest_of_line}"
                    new_lines.append(line)
            with open(f"{outpath}/{category_name}.tsv", 'w') as new_tsv:
                new_tsv.writelines(new_lines)


penman.dump(full_corpus, f"{outpath}/corpus.txt")

if am:
    penman.dump(am_full_corpus, f"{am_path}/full_corpus.txt")

