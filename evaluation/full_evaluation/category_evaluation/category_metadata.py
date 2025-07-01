from copy import copy

from evaluation.full_evaluation.category_evaluation.evaluation_classes import EdgeRecall, NodeRecall, NERecall, \
    NETypeRecall, WordDisambiguationRecall, PPAttachment, EllipsisRecall, SubgraphRecall, ImperativeRecall, \
    ExactMatch
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.novel_corpus.structural_generalization import \
    add_sanity_check_suffix
from evaluation.util import SANITY_CHECK

set_names_with_category_names = [
    ("1. Pragmatic reentrancies", ["pragmatic_coreference_testset", "pragmatic_coreference_winograd"]),
    ("2. Unambiguous reentrancies", ["syntactic_gap_reentrancies", "unambiguous_coreference"]),
    ("3. Structural generalization",
     ["nested_control_and_coordination", "nested_control_and_coordination_sanity_check",
      "multiple_adjectives", "multiple_adjectives_sanity_check",
      "centre_embedding", "centre_embedding_sanity_check",
      "cp_recursion", "cp_recursion_sanity_check",
      "cp_recursion_plus_coreference", "cp_recursion_plus_coreference_sanity_check",
      "cp_recursion_plus_rc", "cp_recursion_plus_rc_sanity_check",
      "cp_recursion_plus_rc_plus_coreference", "cp_recursion_plus_rc_plus_coreference_sanity_check",
      "long_lists", "long_lists_sanity_check"
      ]),
    ("4. Rare and unseen words",
     ["rare_node_labels", "unseen_node_labels", "rare_predicate_senses_excl_01", "unseen_predicate_senses_excl_01",
      "rare_edge_labels_ARG2plus", "unseen_edge_labels_ARG2plus"]),
    ("5. Special entities",
     ["seen_names", "unseen_names", "seen_dates", "unseen_dates", "other_seen_entities", "other_unseen_entities"]),
    ("6. Entity classification and linking",
     ["types_of_seen_named_entities", "types_of_unseen_named_entities", "seen_andor_easy_wiki_links",
      "hard_unseen_wiki_links"]),
    ("7. Lexical disambiguations",
     ["frequent_predicate_senses_incl_01", "word_ambiguities_handcrafted", "word_ambiguities_karidi_et_al_2021"]),
    ("8. Edge attachments", ["pp_attachment", "unbounded_dependencies", "passives", "unaccusatives"]),
    ("9. Non-trivial word-to-node relations", ["ellipsis", "multinode_word_meanings", "imperatives"])
    ]

bunch2subcategory = dict(set_names_with_category_names)

category_name_to_set_class_and_metadata = {
    "pragmatic_coreference_testset": (EdgeRecall, SubcategoryMetadata(
    "pragmatic_coreference_testset",
    "Pragmatic coreference (testset)",
    "reentrancies_pragmatic_filtered.tsv",
    parent_column=4,
    parent_edge_column=5,
    )),
    "pragmatic_coreference_winograd": (EdgeRecall, SubcategoryMetadata(
        "pragmatic_coreference_winograd",
        "Pragmatic coreference (Winograd)",
        "winograd.tsv",
        subcorpus_filename="winograd",
        parent_column=4,
        parent_edge_column=5,
        first_row_is_header=True
    )),
    "syntactic_gap_reentrancies": (EdgeRecall, SubcategoryMetadata(
        "syntactic_gap_reentrancies",
        display_name="Syntactic (gap) reentrancies",
        tsv="reentrancies_syntactic_gap_filtered.tsv",
        parent_column=4,
        parent_edge_column=5
    )),
    "unambiguous_coreference": (EdgeRecall, SubcategoryMetadata(
        "unambiguous_coreference",
        display_name="Unambiguous coreference",
        tsv="reentrancies_unambiguous_coreference_filtered.tsv",
        parent_column=4,
        parent_edge_column=5
    )),
    "nested_control_and_coordination": (ExactMatch, SubcategoryMetadata(
        "nested_control_and_coordination",
        display_name="Nested control and coordination",
        subcorpus_filename="nested_control",
        subtype="structural_generalization",
    )),
    "multiple_adjectives": (ExactMatch, SubcategoryMetadata(
        "multiple_adjectives",
        display_name="Multiple adjectives",
        subcorpus_filename="adjectives",
        subtype="structural_generalization",
    )),
    "centre_embedding": (ExactMatch, SubcategoryMetadata(
        "centre_embedding",
        display_name="Centre embedding",
        subcorpus_filename="centre_embedding",
        subtype="structural_generalization",
    )),
    "cp_recursion": (ExactMatch, SubcategoryMetadata(
        "cp_recursion",
        display_name="CP recursion",
        subcorpus_filename="deep_recursion_basic",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_coreference": (ExactMatch, SubcategoryMetadata(
        "cp_recursion_plus_coreference",
        display_name="CP recursion + coreference",
        subcorpus_filename="deep_recursion_pronouns",
        subtype="structural_generalization",
        extra_subcorpus_filenames=["deep_recursion_3s"]
    )),
    "cp_recursion_plus_rc": (ExactMatch, SubcategoryMetadata(
        "cp_recursion_plus_rc",
        display_name="CP recursion + relative clause (RC)",
        subcorpus_filename="deep_recursion_rc",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_rc_plus_coreference": (ExactMatch, SubcategoryMetadata(
        "cp_recursion_plus_rc_plus_coreference",
        display_name="CP recursion + RC + coreference",
        subcorpus_filename="deep_recursion_rc_contrastive_coref",
        subtype="structural_generalization",
    )),
    "long_lists": (ExactMatch, SubcategoryMetadata(
        "long_lists",
        display_name="Long lists",
        subcorpus_filename="long_lists",
        subtype="structural_generalization",
    )),
    "rare_node_labels": (NodeRecall, SubcategoryMetadata(
        "rare_node_labels",
        display_name="Rare node labels",
        tsv="rare_node_labels_test.tsv",
        use_sense=True,
        run_prerequisites=False)),
    "unseen_node_labels": (NodeRecall, SubcategoryMetadata(
        "unseen_node_labels",
        display_name="Unseen node labels",
        tsv="unseen_node_labels_test_filtered.tsv",
        use_sense=True,
        run_prerequisites=False)),
    "rare_predicate_senses_excl_01": (NodeRecall, SubcategoryMetadata(
        "rare_predicate_senses_excl_01",
        display_name="Rare predicate senses (excl. -01)",
        latex_display_name="Rare predicate senses (excl.~\\nl{-01})",
        tsv="rare_senses_filtered.tsv",
        use_sense=True)),
    "unseen_predicate_senses_excl_01": (NodeRecall, SubcategoryMetadata(
        "unseen_predicate_senses_excl_01",
        display_name="Unseen predicate senses (excl. -01)",
        latex_display_name="Unseen predicate senses (excl.~\\nl{-01})",
        tsv="unseen_senses_new_sentences.tsv",
        subcorpus_filename="unseen_senses_new_sentences",
        use_sense=True,
    )),
    "rare_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(
        "rare_edge_labels_ARG2plus",
        display_name="Rare edge labels (ARG2+)",
        latex_display_name="Rare edge labels (\\nl{ARG2}+)",
        tsv="rare_roles_arg2plus_filtered.tsv",
        use_sense=True
                                  )),
    "unseen_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(
        "unseen_edge_labels_ARG2plus",
        display_name="Unseen edge labels (ARG2+)",
        latex_display_name="Unseen edge labels (\\nl{ARG2}+)",
        tsv="unseen_roles_new_sentences.tsv", use_sense=True,
        subcorpus_filename="unseen_roles_new_sentences"
    )),
    "seen_names": (NERecall, SubcategoryMetadata(
        "seen_names",
        "Seen names",
        tsv="seen_names.tsv",
        subtype="name",
        metric_label="Recall"
    )),
    "unseen_names": (NERecall, SubcategoryMetadata(
        "unseen_names",
        "Unseen names",
        tsv="unseen_names.tsv",
        subtype="name",
        metric_label="Recall"
    )),
    "seen_dates": (NERecall, SubcategoryMetadata(
        "seen_dates",
        "Seen dates",
        tsv="seen_dates.tsv",
        subtype="date-entity",
        metric_label="Recall"
    )),
    "unseen_dates": (NERecall, SubcategoryMetadata(
        "unseen_dates",
        "Unseen dates",
        tsv="unseen_dates.tsv",
        subtype="date-entity",
        metric_label="Recall"
    )),
    "other_seen_entities": (NERecall, SubcategoryMetadata(
        "other_seen_entities",
        "Other seen entities",
        tsv="seen_special_entities.tsv",
        subtype="other",
        metric_label="Recall",
        label_column=3
    )),
    "other_unseen_entities": (NERecall, SubcategoryMetadata(
        "other_unseen_entities",
        "Other unseen entities",
        tsv="unseen_special_entities.tsv",
        subtype="other",
        metric_label="Recall",
        label_column=3
    )),
    "types_of_seen_named_entities": (NETypeRecall, SubcategoryMetadata(
        "types_of_seen_named_entities",
        "Types of seen named entities",
        tsv="seen_ne_types_test.tsv",
    )),
    "types_of_unseen_named_entities": (NETypeRecall, SubcategoryMetadata(
        "types_of_unseen_named_entities",
        "Types of unseen named entities",
        tsv="unseen_ne_types_test.tsv",
    )),
    "seen_andor_easy_wiki_links": (NodeRecall, SubcategoryMetadata(
        "seen_andor_easy_wiki_links",
        "Seen and/or easy wiki links",
        tsv="seen_andor_easy_wiki_test_data.tsv",
        use_sense=True, use_attributes=True, attribute_label=":wiki", metric_label="Recall",
        run_prerequisites=False
    )),
    "hard_unseen_wiki_links": (NodeRecall, SubcategoryMetadata(
        "hard_unseen_wiki_links",
        "Hard unseen wiki links",
        tsv="hard_wiki_test_data.tsv",
        use_sense=True, use_attributes=True, attribute_label=":wiki", metric_label="Recall",
        run_prerequisites=False
    )),
    "frequent_predicate_senses_incl_01": (NodeRecall, SubcategoryMetadata(
        "frequent_predicate_senses_incl_01",
        "Frequent predicate senses (incl. -01)",
        latex_display_name="Frequent predicate senses (incl. ~\\nl{-01})",
        tsv="common_senses_filtered.tsv", use_sense=True, run_prerequisites=True
    )),
    "word_ambiguities_handcrafted": (WordDisambiguationRecall, SubcategoryMetadata(
        "word_ambiguities_handcrafted",
        "Word ambiguities (handcrafted)",
        subcorpus_filename="word_disambiguation",
        subtype="hand-crafted",
    )),
    "word_ambiguities_karidi_et_al_2021": (WordDisambiguationRecall, SubcategoryMetadata(
        "word_ambiguities_karidi_et_al_2021",
        "Word ambiguities (karidi-et-al-2021)",
        latex_display_name="Word ambiguities \cite{karidi-etal-2021-putting}",
        subcorpus_filename="berts_mouth",
        subtype="bert"
    )),
    "pp_attachment": (PPAttachment, SubcategoryMetadata(
        "pp_attachment",
        display_name="PP attachment",
        subcorpus_filename="pp_attachment",
        extra_subcorpus_filenames=["see_with", "read_by", "bought_for", "keep_from", "give_up_in"]
    )),
    "unbounded_dependencies": (EdgeRecall, SubcategoryMetadata(
        "unbounded_dependencies",
        display_name="Unbounded dependencies",
        tsv="unbounded_dependencies.tsv",
        subcorpus_filename="unbounded_dependencies",
        use_sense=False, source_column=2, edge_column=3, target_column=4, first_row_is_header=True)),
    "passives": (EdgeRecall, SubcategoryMetadata(
        "passives",
        display_name="Passives",
        tsv="passives_filtered.tsv", use_sense=True
    )),
    "unaccusatives": (EdgeRecall, SubcategoryMetadata(
        "unaccusatives",
        display_name="Unaccusatives",
        tsv="unaccusatives2_filtered.tsv", use_sense=True
    )),
    "ellipsis": (EllipsisRecall, SubcategoryMetadata(
        "ellipsis",
        display_name="Ellipsis",
        tsv="ellipsis_filtered.tsv"
    )),
    "multinode_word_meanings": (SubgraphRecall, SubcategoryMetadata(
        "multinode_word_meanings",
        "Multinode word meanings",
        tsv="multinode_constants_filtered.tsv",
        metric_label="Recall"
    )),
    "imperatives": (ImperativeRecall, SubcategoryMetadata(
        "imperatives",
        display_name="Imperatives",
        tsv="imperatives_filtered.tsv",
    ))
}

for name in bunch2subcategory["3. Structural generalization"]:
    eval_class, info = category_name_to_set_class_and_metadata[name]
    new_info = copy(info)
    new_name = add_sanity_check_suffix(name)
    new_info.display_name = SANITY_CHECK
    new_info.name = new_name
    new_info.subcorpus_filename = add_sanity_check_suffix(info.subcorpus_filename)
    if new_info.extra_subcorpus_filenames is not None:
        new_info.extra_subcorpus_filenames = [add_sanity_check_suffix(filename) for filename in info.extra_subcorpus_filenames]

    category_name_to_set_class_and_metadata[new_name] = eval_class, new_info



def get_formatted_category_names(names=category_name_to_set_class_and_metadata.keys()):
    return "\n".join(names)


def is_testset_category(info):
    return info.subcorpus_filename is None

#
# for category in category_name_to_set_class_and_metadata:
#     if category != category_name_to_set_class_and_metadata[category][1].name:
#         print(category)