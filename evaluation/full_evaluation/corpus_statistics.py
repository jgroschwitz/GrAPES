# total numbers of data points, split up by source
from copy import copy


# totals

# test set
test_set_evals = [36,41,31,676,117,56,40,1788,910,233,204,237,109,1628,659,2064,277,1654,83,48,33,50,76]
test_set_prereqs = [6,36,41,31,56,40,1628,659,1654,83,48,33,76]

test_set_data_points_recall = sum(test_set_evals)
test_set_data_points_prereqs = sum(test_set_prereqs)
test_set_total = test_set_data_points_recall + test_set_data_points_prereqs

# grammars
long_lists = 1872+408  # number of lists in file
grammar_evals = [50,40,30,100,182,60,70,long_lists,325]
grammar_sanity_checks = [11,11,9,6,24,4,5,111]
grammar_prereqs = 325

grammar_data_points_scores = sum(grammar_evals)
grammar_data_points_sanity_checks = sum(grammar_sanity_checks)
grammar_total = grammar_data_points_scores + grammar_data_points_sanity_checks + grammar_prereqs

# hand written
hand_written = [40,36,47]
hand_written_prereqs = [40,36]

hand_written_data_points_scores = sum(hand_written)
hand_written_data_points_prereqs = sum(hand_written_prereqs)
hand_written_total = hand_written_data_points_prereqs + hand_written_data_points_scores

# external corpora
external_corpora = [40,95,66]
external_corpora_prereqs = [40,66]

external_corpora_data_points_scores = sum(external_corpora)
external_corpora_data_points_prereqs = sum(external_corpora_prereqs)
external_corpora_total = external_corpora_data_points_scores + external_corpora_data_points_prereqs

total_prereqs = hand_written_data_points_prereqs + external_corpora_data_points_prereqs\
                + grammar_prereqs + test_set_data_points_prereqs

total_scores = test_set_data_points_recall + grammar_data_points_scores + hand_written_data_points_scores\
               + external_corpora_data_points_scores

total_prereqs_and_sanity_checks = total_prereqs + grammar_data_points_sanity_checks

total = external_corpora_total + hand_written_total + grammar_total + test_set_total


variables = copy(locals())

for v in variables:
    if not v.startswith("__") and v != "copy":
        print(v, variables[v])


