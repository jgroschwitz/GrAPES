import csv


def main():
    input_files = ["../corpus/reentrancies_labeled_clean.tsv",
                   "../corpus/reentrancies_labeled_clean.tsv",
                   "../corpus/reentrancies_labeled_clean.tsv",
                   "../corpus/unaccusatives2_labelled.tsv",
                   "../corpus/unaccusatives2_labelled.tsv",
                   "../corpus/common_senses_labeled.tsv",
                   "../corpus/multinode_constants_labeled.tsv",
                   "../corpus/imperatives_labeled.tsv",
                   "../corpus/rare_roles_arg2plus_labelled.tsv"]
    output_files = ["../corpus/reentrancies_pragmatic_filtered.tsv",
                    "../corpus/reentrancies_syntactic_gap_filtered.tsv",
                    "../corpus/reentrancies_unambiguous_coreference_filtered.tsv",
                    "../corpus/passives_filtered.tsv",
                    "../corpus/unaccusatives2_filtered.tsv",
                    "../corpus/common_senses_filtered.tsv",
                    "../corpus/multinode_constants_filtered.tsv",
                    "../corpus/imperatives_filtered.tsv",
                    "../corpus/rare_roles_arg2plus_filtered.tsv"]

    keep_labels_pragmatic_reentrancy = ["epithet", "3coref", "repetition",
                                        "pragmatic"]
    # epithet and repetition are kind of the same thing, and both are pragmatic
    # 3coref means pronominal co-reference in third person (is in principle ambiguous)
    keep_labels_syntax_gap_reentrancy = ["N-control", "coord", "control", "secondary-predicate"]
    keep_labels_unambiguous_coreference_reentrancy = ["coref", "self"]
    # coref means first or second person co-reference, which is not ambiguous so we
    # put it here. Maybe call the category "syntactic or forced"?
    # "role of reentrancies in AMR" seems to call N-control pragmatic.
    # We'll treat it as syntactic here.
    # The "syntactic" category has only one sentence, which is a bit iffy, so we leave it out.

    keep_labels_passive = ["passive"]
    keep_labels_unaccusative = ["unaccusative"]

    keep_labels_sets = [keep_labels_pragmatic_reentrancy,
                        keep_labels_syntax_gap_reentrancy,
                        keep_labels_unambiguous_coreference_reentrancy,
                        keep_labels_passive,
                        keep_labels_unaccusative,
                        ["yes"],
                        ["yes"],
                        ["yes"],
                        ["yes"]]

    label_columns = [6,
                     6,
                     6,
                     4,
                     4,
                     2,
                     2,
                     4,
                     4]  # 0-based

    quote_chars = [None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None]  # the quote chars used in the labeled tsv file (input file)

    for input_file, output_file, keep_labels, label_column, quotechar in zip(input_files, output_files,
                                                                             keep_labels_sets, label_columns,
                                                                             quote_chars):
        with open(input_file, "r") as f:
            with open(output_file, "w") as g:
                csvreader = csv.reader(f, delimiter='\t', quotechar=quotechar)
                csvwriter = csv.writer(g, delimiter='\t', quotechar=None, lineterminator="\n")  # always write with quotechar None, because our downstream evaluation code expects it that way
                for row in csvreader:
                    label = row[label_column]
                    if label in keep_labels:
                        csvwriter.writerow(row)


if __name__ == '__main__':
    main()
