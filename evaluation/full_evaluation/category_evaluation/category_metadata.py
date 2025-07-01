from copy import copy

from evaluation.full_evaluation.category_evaluation.evaluation_classes import EdgeRecall, NodeRecall, NERecall, \
    NETypeRecall, WordDisambiguationRecall, PPAttachment, EllipsisRecall, SubgraphRecall, ImperativeRecall, \
    ExactMatch
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.novel_corpus.structural_generalization import \
    add_sanity_check_suffix
from evaluation.util import SANITY_CHECK

# TODO check orders and completeness
bunch2subcategory = {
    "1. Pragmatic Reentrancies": ["pragmatic_coreference_testset", "pragmatic_coreference_winograd"],
    "4. Rare Unseen Nodes Edges": ["rare_node_labels", "unseen_node_labels", "rare_predicate_senses_excl_01",
                         "rare_edge_labels_ARG2plus", "unseen_edge_labels_ARG2plus"],
    "2. Unambiguous Reentrancies": ["syntactic_gap_reentrancies", "unambiguous_coreference"],
    "8. Attachments": ["pp_attachment", "unbounded_dependencies", "passives", "unaccusatives"],
    "6. Entity Classification And Linking": ["seen_andor_easy_wiki_links", "hard_unseen_wiki_links"],
    "5. Names Dates Etc": ["seen_names", "unseen_names", "seen_dates", "unseen_dates", "other_seen_entities",
                           "other_unseen_entities",  "types_of_seen_named_entities", "types_of_unseen_named_entities"],
    "9. Nontrivial Word2Node Relations": ["ellipsis", "multinode_word_meanings", "imperatives"],
    "7. Lexical Disambiguation": ["frequent_predicate_senses_incl_01", "word_ambiguities_handcrafted", "word_ambiguities_karidi_et_al_2021"],
    "3. Structural Generalization": ["nested_control_and_coordination", "multiple_adjectives", "centre_embedding",
                                     "cp_recursion", "cp_recursion_plus_coreference", "cp_recursion_plus_rc",
                                     "cp_recursion_plus_rc_plus_coreference", "long_lists"],
}

def is_grapes_category_with_testset_data(category_info):
    return  category_info.subcorpus_filename == "word_disambiguation"

def is_grapes_category_with_ptb_data(category_info):
    return category_info.subcorpus_filename == "unbounded_dependencies"

def is_copyrighted_data(category_info):
    return is_grapes_category_with_testset_data(category_info) or is_grapes_category_with_ptb_data(category_info)

def is_sanity_check(category_info):
    return category_info.subcorpus_filename.endswith("sanity_check")

category_name_to_set_class_and_metadata = {
    "pragmatic_coreference_testset": (EdgeRecall, SubcategoryMetadata(
        "Pragmatic coreference (testset)",
        "reentrancies_pragmatic_filtered.tsv",
        parent_column=4,
        parent_edge_column=5,
    )),
    "pragmatic_coreference_winograd": (EdgeRecall, SubcategoryMetadata(
        "Pragmatic coreference (Winograd)",
        "winograd.tsv",
        subcorpus_filename="winograd",
        parent_column=4,
        parent_edge_column=5,
        first_row_is_header=True
    )),
    "syntactic_gap_reentrancies": (EdgeRecall, SubcategoryMetadata(display_name="Syntactic (gap) reentrancies",
                                                                   tsv="reentrancies_syntactic_gap_filtered.tsv",
                                                                   parent_column=4,
                                                                   parent_edge_column=5)),

    "unambiguous_coreference": (EdgeRecall, SubcategoryMetadata(display_name="Unambiguous coreference",
                                                                tsv="reentrancies_unambiguous_coreference_filtered.tsv",
                                                                parent_column=4,
                                                                parent_edge_column=5)),
    "nested_control_and_coordination": (ExactMatch, SubcategoryMetadata(
        display_name="Nested control and coordination",
        subcorpus_filename="nested_control",
        subtype="structural_generalization",
    )),
    "multiple_adjectives": (ExactMatch, SubcategoryMetadata(
        display_name="Multiple adjectives",
        subcorpus_filename="adjectives",
        subtype="structural_generalization",
    )),
    "centre_embedding": (ExactMatch, SubcategoryMetadata(
        display_name="Centre embedding",
        subcorpus_filename="centre_embedding",
        subtype="structural_generalization",
    )),
    "cp_recursion": (ExactMatch, SubcategoryMetadata(
        display_name="CP recursion",
        subcorpus_filename="deep_recursion_basic",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_coreference": (ExactMatch, SubcategoryMetadata(
        display_name="CP recursion + coreference",
        subcorpus_filename="deep_recursion_pronouns",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_rc": (ExactMatch, SubcategoryMetadata(
        display_name="CP recursion + relative clause (RC)",
        subcorpus_filename="deep_recursion_rc",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_rc_plus_coreference": (ExactMatch, SubcategoryMetadata(
        display_name="CP recursion + RC + coreference",
        subcorpus_filename="deep_recursion_rc_contrastive_coref",
        subtype="structural_generalization",
    )),
    "long_lists": (ExactMatch, SubcategoryMetadata(
        display_name="Long lists",
        subcorpus_filename="long_lists",
        subtype="structural_generalization",
    )),
    "rare_node_labels": (NodeRecall, SubcategoryMetadata(display_name="Rare node labels",
                                                         tsv="rare_node_labels_test.tsv",
                                                         use_sense=True,
                                                         run_prerequisites=False)),
    "unseen_node_labels": (NodeRecall, SubcategoryMetadata(display_name="Unseen node labels",
                                                           tsv="unseen_node_labels_test_filtered.tsv",
                                                           use_sense=True,
                                                           run_prerequisites=False)),
    "rare_predicate_senses_excl_01": (NodeRecall, SubcategoryMetadata(
        display_name="Rare predicate senses (excl. -01)",
        latex_display_name="Rare predicate senses (excl.~\\nl{-01})",
        tsv="rare_senses_filtered.tsv",
        use_sense=True)),
    "unseen_predicate_senses_excl_01": (NodeRecall, SubcategoryMetadata(
        display_name="Unseen predicate senses (excl. -01)",
        latex_display_name="Unseen predicate senses (excl.~\\nl{-01})",
        tsv="unseen_senses_new_sentences.tsv",
        subcorpus_filename="unseen_senses_new_sentences",
        use_sense=True,
    )),
    "rare_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(
        display_name="Rare edge labels (ARG2+)",
        latex_display_name="Rare edge labels (\\nl{ARG2}+)",
        tsv="rare_roles_arg2plus_filtered.tsv",
        use_sense=True
                                  )),
    "unseen_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(
        display_name="Unseen edge labels (ARG2+)",
        latex_display_name="Unseen edge labels (\\nl{ARG2}+)",
        tsv="unseen_roles_new_sentences.tsv", use_sense=True,
        subcorpus_filename="unseen_roles_new_sentences"
    )),
    "seen_names": (NERecall, SubcategoryMetadata(
        "Seen names",
        tsv="seen_names.tsv",
        subtype="name",
        metric_label="Recall"
    )),
    "unseen_names": (NERecall, SubcategoryMetadata(
        "Unseen names",
        tsv="unseen_names.tsv",
        subtype="name",
        metric_label="Recall"
    )),
    "seen_dates": (NERecall, SubcategoryMetadata(
        "Seen dates",
        tsv="seen_dates.tsv",
        subtype="date-entity",
        metric_label="Recall"
    )),
    "unseen_dates": (NERecall, SubcategoryMetadata(
        "Unseen dates",
        tsv="unseen_dates.tsv",
        subtype="date-entity",
        metric_label="Recall"
    )),
    "other_seen_entities": (NERecall, SubcategoryMetadata(
        "Other seen entities",
        tsv="seen_special_entities.tsv",
        subtype="other",
        metric_label="Recall",
        label_column=3
    )),
    "other_unseen_entities": (NERecall, SubcategoryMetadata(
        "Other unseen entities",
        tsv="unseen_special_entities.tsv",
        subtype="other",
        metric_label="Recall",
        label_column=3
    )),
    "types_of_seen_named_entities": (NETypeRecall, SubcategoryMetadata(
        "Types of seen named entities",
        tsv="seen_ne_types_test.tsv",
    )),
    "types_of_unseen_named_entities": (NETypeRecall, SubcategoryMetadata(
        "Types of unseen named entities",
        tsv="unseen_ne_types_test.tsv",
    )),
    "seen_andor_easy_wiki_links": (NodeRecall, SubcategoryMetadata(
        "Seen and/or easy wiki links",
        tsv="seen_andor_easy_wiki_test_data.tsv",
        use_sense=True, use_attributes=True, attribute_label=":wiki", metric_label="Recall",
        run_prerequisites=False
    )),
    "hard_unseen_wiki_links": (NodeRecall, SubcategoryMetadata(
        "Hard unseen wiki links",
        tsv="hard_wiki_test_data.tsv",
        use_sense=True, use_attributes=True, attribute_label=":wiki", metric_label="Recall",
        run_prerequisites=False
    )),
    "frequent_predicate_senses_incl_01": (NodeRecall, SubcategoryMetadata(
        "Frequent predicate senses (incl. -01)",
        latex_display_name="Frequent predicate senses (incl. ~\\nl{-01})",
        tsv="common_senses_filtered.tsv", use_sense=True, run_prerequisites=True
    )),
    "word_ambiguities_handcrafted": (WordDisambiguationRecall, SubcategoryMetadata(
        "Word ambiguities (handcrafted)",
        subcorpus_filename="word_disambiguation",
        subtype="hand-crafted",
    )),
    "word_ambiguities_karidi_et_al_2021": (WordDisambiguationRecall, SubcategoryMetadata(
        "Word ambiguities (karidi-et-al-2021)",
        latex_display_name="Word ambiguities \cite{karidi-etal-2021-putting}",
        subcorpus_filename="berts_mouth",
        subtype="bert"
    )),
    "pp_attachment": (PPAttachment, SubcategoryMetadata(
        display_name="PP attachment",
        subcorpus_filename="pp_attachment",
    )),
    "unbounded_dependencies": (EdgeRecall, SubcategoryMetadata(
        display_name="Unbounded dependencies",
        tsv="unbounded_dependencies.tsv",
        subcorpus_filename="unbounded_dependencies",
        use_sense=False, source_column=2, edge_column=3, target_column=4, first_row_is_header=True)),
    "passives": (EdgeRecall, SubcategoryMetadata(
        display_name="Passives",
        tsv="passives_filtered.tsv", use_sense=True
    )),
    "unaccusatives": (EdgeRecall, SubcategoryMetadata(
        display_name="Unaccusatives",
        tsv="unaccusatives2_filtered.tsv", use_sense=True
    )),
    "ellipsis": (EllipsisRecall, SubcategoryMetadata(
        display_name="Ellipsis",
        tsv="ellipsis_filtered.tsv"
    )),
    "multinode_word_meanings": (SubgraphRecall, SubcategoryMetadata(
        "Multinode word meanings",
        tsv="multinode_constants_filtered.tsv",
        metric_label="Recall"
    )),
    "imperatives": (ImperativeRecall, SubcategoryMetadata(
        display_name="Imperatives",
        tsv="imperatives_filtered.tsv",
    ))
}

for name in bunch2subcategory["3. Structural Generalization"]:
    eval_class, info = category_name_to_set_class_and_metadata[name]
    new_info = copy(info)
    new_name = add_sanity_check_suffix(name)
    new_info.display_name = SANITY_CHECK
    new_info.subcorpus_filename = add_sanity_check_suffix(info.subcorpus_filename)
    category_name_to_set_class_and_metadata[new_name] = eval_class, new_info


def get_formatted_category_names(names=category_name_to_set_class_and_metadata.keys()):
    return "\n".join(names)


def is_testset_category(info):
    return info.subcorpus_filename is None
