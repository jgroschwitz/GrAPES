from copy import copy

from evaluation.full_evaluation.category_evaluation.category_evaluation import STRUC_GEN
from evaluation.full_evaluation.category_evaluation.misc_evaluation import (EllipsisRecall, SubgraphRecall,
                                                                               ImperativeRecall, TARGET)
from evaluation.full_evaluation.category_evaluation.exact_match import ExactMatch
from evaluation.full_evaluation.category_evaluation.list_accuracy import ListAccuracy
from evaluation.full_evaluation.category_evaluation.named_entities import NETypeRecall, NERecall
from evaluation.full_evaluation.category_evaluation.node_recall import NodeRecall
from evaluation.full_evaluation.category_evaluation.edge_recall import EdgeRecall
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata, is_sanity_check
from evaluation.full_evaluation.category_evaluation.word_disambiguation import WordDisambiguationBertsMouth, \
    WordDisambiguationHandcrafted
from evaluation.full_evaluation.category_evaluation.pp_attachment import PPAttachment
from evaluation.util import SANITY_CHECK


bunch_number2name = {
    1: "Pragmatic reentrancies",
    2: "Unambiguous reentrancies",
    3: "Structural generalization",
    4: "Rare and unseen words",
    5: "Special entities",
    6: "Entity classification and linking",
    7: "Lexical disambiguations",
    8: "Edge attachments",
    9: "Non-trivial word-to-node relations"
}

# bunch_names = ['Pragmatic reentrancies', 'Unambiguous reentrancies', 'Structural generalization', 'Rare and unseen words', 'Special entities', 'Entity classification and linking', 'Lexical disambiguations', 'Edge attachments', 'Non-trivial word-to-node relations']

bunch2subcategory = {'Pragmatic reentrancies': ['pragmatic_coreference_testset', 'pragmatic_coreference_winograd'],
                     'Unambiguous reentrancies': ['syntactic_gap_reentrancies', 'unambiguous_coreference'],
                     'Structural generalization': [
                         'nested_control_and_coordination', 'nested_control_and_coordination_sanity_check',
                         'multiple_adjectives', 'multiple_adjectives_sanity_check',
                         'centre_embedding', 'centre_embedding_sanity_check',
                         'cp_recursion', 'cp_recursion_sanity_check',
                         'cp_recursion_plus_coreference', 'cp_recursion_plus_coreference_sanity_check',
                         'cp_recursion_plus_rc', 'cp_recursion_plus_rc_sanity_check',
                         'cp_recursion_plus_rc_plus_coreference', 'cp_recursion_plus_rc_plus_coreference_sanity_check',
                         'long_lists', 'long_lists_sanity_check'],
                     'Rare and unseen words': ['rare_node_labels', 'unseen_node_labels',
                                               'rare_predicate_senses_excl_01', 'unseen_predicate_senses_excl_01',
                                               'rare_edge_labels_ARG2plus', 'unseen_edge_labels_ARG2plus'],
                     'Special entities': ['seen_names', 'unseen_names', 'seen_dates', 'unseen_dates',
                                          'other_seen_entities', 'other_unseen_entities'],
                     'Entity classification and linking': [
                         'types_of_seen_named_entities', 'types_of_unseen_named_entities',
                         'seen_andor_easy_wiki_links', 'hard_unseen_wiki_links'],
                     'Lexical disambiguations': ['frequent_predicate_senses_incl_01',
                                                 'word_ambiguities_handcrafted', 'word_ambiguities_karidi_et_al_2021'],
                     'Edge attachments': ['pp_attachment', 'unbounded_dependencies', 'passives', 'unaccusatives'],
                     'Non-trivial word-to-node relations': ['ellipsis', 'multinode_word_meanings', 'imperatives']}



def get_bunch_name_for_number(n):
    return bunch_number2name[n]

def get_bunch_categories_for_number(n):
    return bunch2subcategory[get_bunch_name_for_number(n)]

def get_bunch_categories_for_name(name):
    return bunch2subcategory[name]


# set_names_with_category_names = [
#     ("1. Pragmatic reentrancies", ["pragmatic_coreference_testset", "pragmatic_coreference_winograd"]),
#     ("2. Unambiguous reentrancies", ["syntactic_gap_reentrancies", "unambiguous_coreference"]),
#     ("3. Structural generalization",
#      ["nested_control_and_coordination", "nested_control_and_coordination_sanity_check",
#       "multiple_adjectives", "multiple_adjectives_sanity_check",
#       "centre_embedding", "centre_embedding_sanity_check",
#       "cp_recursion", "cp_recursion_sanity_check",
#       "cp_recursion_plus_coreference", "cp_recursion_plus_coreference_sanity_check",
#       "cp_recursion_plus_rc", "cp_recursion_plus_rc_sanity_check",
#       "cp_recursion_plus_rc_plus_coreference", "cp_recursion_plus_rc_plus_coreference_sanity_check",
#       "long_lists", "long_lists_sanity_check"
#       ]),
#     ("4. Rare and unseen words",
#      ["rare_node_labels", "unseen_node_labels", "rare_predicate_senses_excl_01", "unseen_predicate_senses_excl_01",
#       "rare_edge_labels_ARG2plus", "unseen_edge_labels_ARG2plus"]),
#     ("5. Special entities",
#      ["seen_names", "unseen_names", "seen_dates", "unseen_dates", "other_seen_entities", "other_unseen_entities"]),
#     ("6. Entity classification and linking",
#      ["types_of_seen_named_entities", "types_of_unseen_named_entities", "seen_andor_easy_wiki_links",
#       "hard_unseen_wiki_links"]),
#     ("7. Lexical disambiguations",
#      ["frequent_predicate_senses_incl_01", "word_ambiguities_handcrafted", "word_ambiguities_karidi_et_al_2021"]),
#     ("8. Edge attachments", ["pp_attachment", "unbounded_dependencies", "passives", "unaccusatives"]),
#     ("9. Non-trivial word-to-node relations", ["ellipsis", "multinode_word_meanings", "imperatives"])
#     ]
#
# bunch2subcategory = dict(set_names_with_category_names)

category_name_to_set_class_and_metadata = {
    "pragmatic_coreference_testset": (EdgeRecall, SubcategoryMetadata(
    "pragmatic_coreference_testset",
    "Pragmatic coreference (testset)",
        1,
    tsv="reentrancies_pragmatic_filtered.tsv",
    parent_column=4,
    parent_edge_column=5, metric_label="Edge Recall"
    )),
    "pragmatic_coreference_winograd": (EdgeRecall, SubcategoryMetadata(
        "pragmatic_coreference_winograd",
        "Pragmatic coreference (Winograd)",
        1,
        "winograd.tsv",
        subcorpus_filename="winograd",
        parent_column=4,
        parent_edge_column=5,
        first_row_is_header=True, metric_label="Edge Recall"
    )),
    "syntactic_gap_reentrancies": (EdgeRecall, SubcategoryMetadata(
        "syntactic_gap_reentrancies",
        display_name="Syntactic (gap) reentrancies",
        bunch=2,
        tsv="reentrancies_syntactic_gap_filtered.tsv",
        parent_column=4,
        parent_edge_column=5, metric_label="Edge Recall"
    )),
    "unambiguous_coreference": (EdgeRecall, SubcategoryMetadata(
        "unambiguous_coreference",
        display_name="Unambiguous coreference",
        bunch=2,
        tsv="reentrancies_unambiguous_coreference_filtered.tsv",
        parent_column=4,
        parent_edge_column=5, metric_label="Edge Recall"
    )),
    "nested_control_and_coordination": (ExactMatch, SubcategoryMetadata(
        "nested_control_and_coordination",
        display_name="Nested control and coordination",
        bunch=3,
        subcorpus_filename="nested_control",
        subtype=STRUC_GEN,
        run_prerequisites=False,
        metric_label="Exact Match"
    )),
    "multiple_adjectives": (ExactMatch, SubcategoryMetadata(
        "multiple_adjectives",
        display_name="Multiple adjectives",
        bunch=3,
        subcorpus_filename="adjectives",
        subtype=STRUC_GEN,
        run_prerequisites=False,
        metric_label="Exact Match"
    )),
    "centre_embedding": (ExactMatch, SubcategoryMetadata(
        "centre_embedding",
        display_name="Centre embedding",
        bunch=3,
        subcorpus_filename="centre_embedding",
        subtype=STRUC_GEN,
        run_prerequisites=False,
        metric_label="Exact Match"
    )),
    "cp_recursion": (ExactMatch, SubcategoryMetadata(
        "cp_recursion",
        display_name="CP recursion",
        bunch=3,
        subcorpus_filename="deep_recursion_basic",
        subtype=STRUC_GEN,
        run_prerequisites=False,
        metric_label="Exact Match"
    )),
    "cp_recursion_plus_coreference": (ExactMatch, SubcategoryMetadata(
        "cp_recursion_plus_coreference",
        display_name="CP recursion + coreference",
        bunch=3,
        subcorpus_filename="deep_recursion_pronouns",
        subtype=STRUC_GEN,
        extra_subcorpus_filenames=["deep_recursion_3s"],
        run_prerequisites=False,
        metric_label="Exact Match"
    )),
    "cp_recursion_plus_rc": (ExactMatch, SubcategoryMetadata(
        "cp_recursion_plus_rc",
        display_name="CP recursion + relative clause (RC)",
        bunch=3,
        subcorpus_filename="deep_recursion_rc",
        subtype=STRUC_GEN,
        run_prerequisites=False,
        metric_label="Exact Match"
    )),
    "cp_recursion_plus_rc_plus_coreference": (ExactMatch, SubcategoryMetadata(
        "cp_recursion_plus_rc_plus_coreference",
        display_name="CP recursion + RC + coreference",
        bunch=3,
        subcorpus_filename="deep_recursion_rc_contrastive_coref",
        subtype=STRUC_GEN,
        run_prerequisites=False,
        metric_label="Exact Match"
    )),
    "long_lists": (ListAccuracy, SubcategoryMetadata(
        "long_lists",
        display_name="Long lists",
        bunch=3,
        subcorpus_filename="long_lists",
        metric_label="Conjunct recall",
        subtype = STRUC_GEN,
        run_prerequisites=False
    )),
    "rare_node_labels": (NodeRecall, SubcategoryMetadata(
        "rare_node_labels",
        display_name="Rare node labels",
        bunch=4,
        tsv="rare_node_labels_test.tsv",
        use_sense=True,
        run_prerequisites=False, metric_label="Label Recall"
    )),
    "unseen_node_labels": (NodeRecall, SubcategoryMetadata(
        "unseen_node_labels",
        display_name="Unseen node labels",
        bunch=4,
        tsv="unseen_node_labels_test_filtered.tsv",
        use_sense=True, metric_label="Label Recall",
        run_prerequisites=False)),
    "rare_predicate_senses_excl_01": (NodeRecall, SubcategoryMetadata(
        "rare_predicate_senses_excl_01",
        display_name="Rare predicate senses (excl. -01)",
        bunch=4,
        latex_display_name="Rare predicate senses (excl.~\\nl{-01})",
        tsv="rare_senses_filtered.tsv", metric_label="Label Recall",
        use_sense=True)),
    "unseen_predicate_senses_excl_01": (NodeRecall, SubcategoryMetadata(
        "unseen_predicate_senses_excl_01",
        display_name="Unseen predicate senses (excl. -01)",
        bunch=4,
        latex_display_name="Unseen predicate senses (excl.~\\nl{-01})",
        tsv="unseen_senses_new_sentences.tsv",
        subcorpus_filename="unseen_senses_new_sentences",
        use_sense=True, metric_label="Label Recall",

    )),
    "rare_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(
        "rare_edge_labels_ARG2plus",
        display_name="Rare edge labels (ARG2+)",
        bunch=4,
        latex_display_name="Rare edge labels (\\nl{ARG2}+)",
        tsv="rare_roles_arg2plus_filtered.tsv",
        use_sense=True, metric_label="Edge Recall"
                                  )),
    "unseen_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(
        "unseen_edge_labels_ARG2plus",
        display_name="Unseen edge labels (ARG2+)",
        bunch=4,
        latex_display_name="Unseen edge labels (\\nl{ARG2}+)",
        tsv="unseen_roles_new_sentences.tsv",
        use_sense=True,
        subcorpus_filename="unseen_roles_new_sentences", metric_label="Edge Recall"
    )),
    "seen_names": (NERecall, SubcategoryMetadata(
        "seen_names",
        "Seen names",
        bunch=5,
        tsv="seen_names.tsv",
        subtype="name",
        run_prerequisites=False
    )),
    "unseen_names": (NERecall, SubcategoryMetadata(
        "unseen_names",
        "Unseen names",
        bunch=5,
        tsv="unseen_names.tsv",
        subtype="name",
        run_prerequisites=False
    )),
    "seen_dates": (NERecall, SubcategoryMetadata(
        "seen_dates",
        "Seen dates",
        bunch=5,
        tsv="seen_dates.tsv",
        subtype="date-entity",
        run_prerequisites=False
    )),
    "unseen_dates": (NERecall, SubcategoryMetadata(
        "unseen_dates",
        "Unseen dates",
        bunch=5,
        tsv="unseen_dates.tsv",
        subtype="date-entity",
        run_prerequisites=False
    )),
    "other_seen_entities": (NERecall, SubcategoryMetadata(
        "other_seen_entities",
        "Other seen entities",
        bunch=5,
        tsv="seen_special_entities.tsv",
        subtype="other",
        label_column=3,
        run_prerequisites=False

    )),
    "other_unseen_entities": (NERecall, SubcategoryMetadata(
        "other_unseen_entities",
        "Other unseen entities",
        bunch=5,
        tsv="unseen_special_entities.tsv",
        subtype="other",
        label_column=3,
        run_prerequisites=False

    )),
    "types_of_seen_named_entities": (NETypeRecall, SubcategoryMetadata(
        "types_of_seen_named_entities",
        "Types of seen named entities",
        bunch=6,
        tsv="seen_ne_types_test.tsv",
    )),
    "types_of_unseen_named_entities": (NETypeRecall, SubcategoryMetadata(
        "types_of_unseen_named_entities",
        "Types of unseen named entities",
        bunch=6,
        tsv="unseen_ne_types_test.tsv",
    )),
    "seen_andor_easy_wiki_links": (NodeRecall, SubcategoryMetadata(
        "seen_andor_easy_wiki_links",
        "Seen and/or easy wiki links",
        bunch=6,
        tsv="seen_andor_easy_wiki_test_data.tsv",
        use_sense=True, use_attributes=True, attribute_label=":wiki",
        run_prerequisites=False
    )),
    "hard_unseen_wiki_links": (NodeRecall, SubcategoryMetadata(
        "hard_unseen_wiki_links",
        "Hard unseen wiki links",
        bunch=6,
        tsv="hard_wiki_test_data.tsv",
        use_sense=True, use_attributes=True, attribute_label=":wiki",
        run_prerequisites=False
    )),
    "frequent_predicate_senses_incl_01": (NodeRecall, SubcategoryMetadata(
        "frequent_predicate_senses_incl_01",
        "Frequent predicate senses (incl. -01)",
        bunch=7,
        latex_display_name="Frequent predicate senses (incl. ~\\nl{-01})",
        tsv="common_senses_filtered.tsv", use_sense=True, run_prerequisites=True,
    )),
    "word_ambiguities_handcrafted": (WordDisambiguationHandcrafted, SubcategoryMetadata(
        "word_ambiguities_handcrafted",
        "Word ambiguities (handcrafted)",
        bunch=7,
        subcorpus_filename="word_disambiguation",
        subtype="hand-crafted",
        run_prerequisites=False
    )),
    "word_ambiguities_karidi_et_al_2021": (WordDisambiguationBertsMouth, SubcategoryMetadata(
        "word_ambiguities_karidi_et_al_2021",
        "Word ambiguities (karidi-et-al-2021)",
        bunch=7,
        latex_display_name="Word ambiguities \cite{karidi-etal-2021-putting}",
        subcorpus_filename="berts_mouth",
        subtype="bert",
        run_prerequisites=False
    )),
    "pp_attachment": (PPAttachment, SubcategoryMetadata(
        "pp_attachment",
        bunch=8,
        display_name="PP attachment",
        subcorpus_filename="pp_attachment",
        extra_subcorpus_filenames=["see_with", "read_by", "bought_for", "keep_from", "give_up_in"],
        metric_label="Edge Recall"
    )),
    "unbounded_dependencies": (EdgeRecall, SubcategoryMetadata(
        "unbounded_dependencies",
        display_name="Unbounded dependencies",
        bunch=8,
        tsv="unbounded_dependencies.tsv",
        subcorpus_filename="unbounded_dependencies",
        use_sense=False, source_column=2, edge_column=3, target_column=4,
        first_row_is_header=True, metric_label="Edge Recall"
    )),
    "passives": (EdgeRecall, SubcategoryMetadata(
        "passives",
        display_name="Passives",
        bunch=8,
        tsv="passives_filtered.tsv", use_sense=True, metric_label="Edge Recall"
    )),
    "unaccusatives": (EdgeRecall, SubcategoryMetadata(
        "unaccusatives",
        display_name="Unaccusatives",
        bunch=8,
        tsv="unaccusatives2_filtered.tsv", use_sense=True
    )),
    "ellipsis": (EllipsisRecall, SubcategoryMetadata(
        "ellipsis",
        display_name="Ellipsis",
        bunch=9,
        tsv="ellipsis_filtered.tsv"
    )),
    "multinode_word_meanings": (SubgraphRecall, SubcategoryMetadata(
        "multinode_word_meanings",
        "Multinode word meanings",
        bunch=9,
        tsv="multinode_constants_filtered.tsv",
        metric_label="Recall",
        run_prerequisites=False
    )),
    "imperatives": (ImperativeRecall, SubcategoryMetadata(
        "imperatives",
        display_name="Imperatives",
        bunch=9,
        tsv="imperatives_filtered.tsv",
        additional_fields=[TARGET]
    ))
}


def add_sanity_check_suffix(filename):
    return f"{filename}_sanity_check"

new_ones = {}
for name in bunch2subcategory["Structural generalization"]:
    try:
        eval_class, info = category_name_to_set_class_and_metadata[name]
        new_info = copy(info)
        new_name = add_sanity_check_suffix(name)
        new_info.display_name = SANITY_CHECK
        new_info.name = new_name
        new_info.subcorpus_filename = add_sanity_check_suffix(info.subcorpus_filename)
        if new_info.extra_subcorpus_filenames is not None:
            new_info.extra_subcorpus_filenames = [add_sanity_check_suffix(filename) for filename in info.extra_subcorpus_filenames]
        if info.name == "long_lists":
            new_info.metric_label = "Exact Match"
            eval_class = ExactMatch

        new_ones[new_name] = eval_class, new_info
    except KeyError:
        continue

category_name_to_set_class_and_metadata.update(new_ones)


def get_formatted_category_names_by_main_file():
    grapes, testset = get_categories_by_main_file()
    header = "AMR 3.0 testset category names:"
    ret = f"\n{'-'*len(header)}\n{header}\n{'-'*len(header)}\n"
    ret += "\n".join(testset)
    header = "GrAPES category names:"
    ret += f"\n{'-'*len(header)}\n{header}\n{'-'*len(header)}\n"
    ret += "\n".join(grapes)
    return ret


def get_categories_by_main_file():
    grapes = []
    testset = []
    for bunch in sorted(bunch2subcategory.keys()):
        for category in bunch2subcategory[bunch]:
            if is_testset_category(category_name_to_set_class_and_metadata[category][1]):
                testset.append(category)
            else:
                grapes.append(category)
    return grapes, testset


def get_formatted_category_names(names=category_name_to_set_class_and_metadata.keys()):
    return "\n".join(names)


def is_testset_category(info):
    return info.subcorpus_filename is None

# with open("categories.txt", "w") as f:
#     f.write(get_formatted_category_names_by_main_file())

# print(list(bunch_number2name.values()))

print(bunch2subcategory)