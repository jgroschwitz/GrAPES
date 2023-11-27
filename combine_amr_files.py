"""
Reads in all txt files in the corpus folder and adds them to corpus.txt
with updated IDs
"""

import os
import sys

import penman

full_corpus = []

for corpus_file in os.listdir("corpus"):
    if corpus_file.endswith(".txt") and corpus_file != "corpus.txt":  # corpus_file == "winograd.txt":
        # print("file", corpus_file, file=sys.stderr)
        category = penman.load(f"corpus/{corpus_file}")
        changed = False
        category_name = corpus_file[:-4]
        if corpus_file == "word_disambiguation.txt":
            word_disambiguation = []
            for i, entry in enumerate(category):
                # copy old ID to supplementary info
                entry.metadata["suppl"] = entry.metadata["id"]
                entry.metadata["id"] = f"{category_name}_{i}"
                if entry.metadata["suppl"].startswith("word_disambiguation"):
                    word_disambiguation.append(entry)

            full_corpus += word_disambiguation
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
            if changed:
                print("ids changed in", corpus_file)

        if corpus_file in ["unseen_senses_new_sentences.txt",
                           "unseen_roles_new_sentences.txt",
                           "winograd.txt"]:
            # update the tsv file with new IDs
            with open(f"corpus/{category_name}_old_ids.tsv", 'r') as tsv:
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
            with open(f"corpus/{category_name}.tsv", 'w') as new_tsv:
                new_tsv.writelines(new_lines)


penman.dump(full_corpus, "corpus/corpus.txt")
