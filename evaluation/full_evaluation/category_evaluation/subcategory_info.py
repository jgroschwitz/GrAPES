from dataclasses import dataclass
from typing import List

from penman import Graph


# from evaluation.full_evaluation.category_evaluation.i_pragmatic_reentrancies import PragmaticReentrancies


@dataclass
class SubcategoryMetadata:
    """
    Stores info about each subcategory
    """
    display_name: str
    tsv: str or None = None
    subcorpus_filename: str or None = None
    other_display_name: str or None = None
    other_subcorpus_filename: str or None = None
    graph_id_column: int = 0
    use_sense: bool = False
    use_sense_prereq = False
    first_row_is_header: bool = False
    # for nodes
    use_attributes: bool = False
    attribute_label: str or None = None
    metric_label: str = "Label recall"
    run_prerequisites: bool = True
    # for edges
    source_column: int or None = 1
    edge_column: int or None = 2
    target_column: int or None = 3
    parent_column: int or None = None
    parent_edge_column: int or None = None
    # for named entities
    entity_type: str or None = None


# category_name_to_subcategory_info = {
#     "pragmatic_coreference_testset": SubcategoryMetadata(
#         "Pragmatic coreference (testset)",
#         "reentrancies_pragmatic_filtered.tsv",
#         parent_column=4,
#         parent_edge_column=5,
#     ),
#     "pragmatic_coreference_winograd": SubcategoryMetadata(
#         "Pragmatic coreference (Winograd)",
#         "winograd.tsv",
#         parent_column=4,
#         parent_edge_column=5,
#         first_row_is_header=True
#     ),
#     # "syntactic_gap_reentrancies": (UnambiguousReentrancies, UnambiguousReentrancies.compute_syntactic_gap_results),
#     # "unambiguous_coreference": (UnambiguousReentrancies, UnambiguousReentrancies.compute_unambiguous_coreference_results),
#     # "nested_control_and_coordination": (StructuralGeneralization, StructuralGeneralization.computed_nested_control_and_coordination_results),
#     # "nested_control_and_coordination_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_nested_control_and_coordination_sanity_check_results),
#     # "multiple_adjectives": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_results),
#     # "multiple_adjectives_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_sanity_check_results),
#     # "centre_embedding": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_results),
#     # "centre_embedding_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_sanity_check_results),
#     # "cp_recursion": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_results),
#     # "cp_recursion_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_sanity_check_results),
#     # "cp_recursion_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_results),
#     # "cp_recursion_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_sanity_check_results),
#     # "cp_recursion_plus_rc": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_results),
#     # "cp_recursion_plus_rc_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_sanity_check_results),
#     # "cp_recursion_plus_rc_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_results),
#     # "cp_recursion_plus_rc_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_sanity_check_results),
#     # "long_lists": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_results),
#     # "long_lists_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_sanity_check_results),
#     # "rare_node_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_node_label_results),
#     # "unseen_node_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_node_label_results),
#     # "rare_predicate_senses_excl_01": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_sense_results),
#     # "unseen_predicate_senses_excl_01": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_sense_results),
#     # "rare_edge_labels_ARG2plus": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_edge_label_results),
#     # "unseen_edge_labels_ARG2plus": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_edge_label_results),
#     # "seen_names": (NamesDatesEtc, NamesDatesEtc.compute_seen_names_results),
#     # "unseen_names": (NamesDatesEtc, NamesDatesEtc.compute_unseen_names_results),
#     # "seen_dates": (NamesDatesEtc, NamesDatesEtc.compute_seen_dates_results),
#     # "unseen_dates": (NamesDatesEtc, NamesDatesEtc.compute_unseen_dates_results),
#     # "other_seen_entities": (NamesDatesEtc, NamesDatesEtc.compute_seen_special_entities_results),
#     # "other_unseen_entities": (NamesDatesEtc, NamesDatesEtc.compute_unseen_special_entities_results),
#     # "types_of_seen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_ne_types_results),
#     # "types_of_unseen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_unseen_ne_types_results),
#     # "seen_andor_easy_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_andor_easy_wiki_results),
#     # "hard_unseen_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_hard_wiki_results),
#     # "frequent_predicate_senses_incl_01": (LexicalDisambiguation, LexicalDisambiguation.compute_common_senses_results),
#     # "word_ambiguities_handcrafted": (LexicalDisambiguation, LexicalDisambiguation.compute_grapes_word_disambiguation_results),
#     # "word_ambiguities_karidi_et_al_2021": (LexicalDisambiguation, LexicalDisambiguation.compute_berts_mouth_results),
#     # "pp_attachment": (Attachments, Attachments.compute_pp_results),
#     # "unbounded_dependencies": (Attachments, Attachments.compute_unbounded_results),
#     # "passives": (Attachments, Attachments.compute_passive_results),
#     # "unaccusatives": (Attachments, Attachments.compute_unaccusative_results),
#     # "ellipsis": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_ellipsis_results),
#     # "multinode_word_meanings": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_multinode_constants_results),
#     # "imperatives": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_imperative_results)
# }
