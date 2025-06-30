from copy import copy

from evaluation.full_evaluation.category_evaluation.evaluation_classes import EdgeRecall, NodeRecall, NERecall, \
    NETypeRecall, WordDisambiguationRecall, PPAttachment, EllipsisRecall, SubgraphRecall, ImperativeRecall, \
    StructuralGeneralisation, ExactMatch
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.structural_generalization import \
    structural_generalization_corpus_names as structural_generalization_corpus_names, add_sanity_check_suffix

SANITY_CHECK = "Sanity check"

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


category_name_to_print_name = {
    "pragmatic_coreference_testset": "Pragmatic coreference (testset)",
    "pragmatic_coreference_winograd": "Pragmatic coreference (Winograd)",
    "syntactic_gap_reentrancies": "Syntactic (gap) reentrancies",
    "unambiguous_coreference": "Unambiguous coreference",
    "nested_control_and_coordination": "Nested control and coordination",
    "nested_control_and_coordination_sanity_check": "Sanity check",
    "multiple_adjectives": "Multiple adjectives",
    "multiple_adjectives_sanity_check": "Sanity check",
    "centre_embedding": "Centre embedding",
    "centre_embedding_sanity_check": "Sanity check",
    "cp_recursion": "CP recursion",
    "cp_recursion_sanity_check": "Sanity check",
    "cp_recursion_plus_coreference": "CP recursion + coreference",
    "cp_recursion_plus_coreference_sanity_check": "Sanity check",
    "cp_recursion_plus_rc": "CP recursion + relative clause (RC)",
    "cp_recursion_plus_rc_sanity_check": "Sanity check",
    "cp_recursion_plus_rc_plus_coreference": "CP recursion + RC + coreference",
    "cp_recursion_plus_rc_plus_coreference_sanity_check": "Sanity check",
    "long_lists": "Long lists",
    "long_lists_sanity_check": "Sanity check",
    "rare_node_labels": "Rare node labels",
    "unseen_node_labels": "Unseen node labels",
    "rare_predicate_senses_excl_01": "Rare predicate senses (excl. -01)",
    "unseen_predicate_senses_excl_01": "Unseen predicate senses (excl. -01)",
    "rare_edge_labels_ARG2plus": "Rare edge labels (ARG2+)",
    "unseen_edge_labels_ARG2plus": "Unseen edge labels (ARG2+)",
    "seen_names": "Seen names",
    "unseen_names": "Unseen names",
    "seen_dates": "Seen dates",
    "unseen_dates": "Unseen dates",
    "other_seen_entities": "Other seen entities",
    "other_unseen_entities": "Other unseen entities",
    "types_of_seen_named_entities": "Types of seen named entities",
    "types_of_unseen_named_entities": "Types of unseen named entities",
    "seen_andor_easy_wiki_links": "Seen and/or easy wiki links",
    "hard_unseen_wiki_links": "Hard unseen wiki links",
    "frequent_predicate_senses_incl_01": "Frequent predicate senses (incl. -01)",
    "word_ambiguities_handcrafted": "Word ambiguities (handcrafted)",
    "word_ambiguities_karidi_et_al_2021": "Word ambiguities (Karidi et al., 2021)",
    "pp_attachment": "PP attachment",
    "unbounded_dependencies": "Unbounded dependencies",
    "passives": "Passives",
    "unaccusatives": "Unaccusatives",
    "ellipsis": "Ellipsis",
    "multinode_word_meanings": "Multinode word meanings",
    "imperatives": "Imperatives"
}





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
        display_name=category_name_to_print_name["nested_control_and_coordination"],
        subcorpus_filename="nested_control",
        subtype="structural_generalization",
    )),
    "multiple_adjectives": (ExactMatch, SubcategoryMetadata(
        display_name=category_name_to_print_name["multiple_adjectives"],
        subcorpus_filename="adjectives",
        subtype="structural_generalization",
    )),
    "centre_embedding": (ExactMatch, SubcategoryMetadata(
        display_name=category_name_to_print_name["centre_embedding"],
        subcorpus_filename="centre_embedding",
        subtype="structural_generalization",
    )),
    "cp_recursion": (ExactMatch, SubcategoryMetadata(
        display_name=category_name_to_print_name["cp_recursion"],
        subcorpus_filename="deep_recursion_basic",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_coreference": (ExactMatch, SubcategoryMetadata(
        display_name=category_name_to_print_name["cp_recursion_plus_coreference"],
        subcorpus_filename="deep_recursion_pronouns",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_rc": (ExactMatch, SubcategoryMetadata(
        display_name=category_name_to_print_name["cp_recursion_plus_rc"],
        subcorpus_filename="deep_recursion_rc",
        subtype="structural_generalization",
    )),
    "cp_recursion_plus_rc_plus_coreference": (ExactMatch, SubcategoryMetadata(
        display_name=category_name_to_print_name["cp_recursion_plus_rc_plus_coreference"],
        subcorpus_filename="deep_recursion_rc_contrastive_coref",
        subtype="structural_generalization",
    )),
    "long_lists": (ExactMatch, SubcategoryMetadata(
        display_name=category_name_to_print_name["long_lists"],
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
        display_name=category_name_to_print_name["unseen_predicate_senses_excl_01"],
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
        metric_label="Recall"
    )),
    "other_unseen_entities": (NERecall, SubcategoryMetadata(
        "Other unseen entities",
        tsv="unseen_special_entities.tsv",
        subtype="other",
        metric_label="Recall"
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
        use_sense=True, use_attributes=True, attribute_label=":wiki", metric_label="Recall"
    )),
    "hard_unseen_wiki_links": (NodeRecall, SubcategoryMetadata(
        "Hard unseen wiki links",
        tsv="hard_wiki_test_data.tsv",
        use_sense=True, use_attributes=True, attribute_label=":wiki", metric_label="Recall"
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

