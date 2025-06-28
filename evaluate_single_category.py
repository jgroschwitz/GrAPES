import argparse
from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.edge_recall_evaluation import EdgeRecall, NodeRecall
from evaluation.full_evaluation.category_evaluation.i_pragmatic_reentrancies import PragmaticReentrancies
from evaluation.full_evaluation.category_evaluation.ii_unambiguous_reentrancies import UnambiguousReentrancies
from evaluation.full_evaluation.category_evaluation.iii_structural_generalization import StructuralGeneralization
from evaluation.full_evaluation.category_evaluation.iv_rare_unseen_nodes_edges import RareUnseenNodesEdges
from evaluation.full_evaluation.category_evaluation.ix_nontrivial_word2node_relations import \
    NontrivialWord2NodeRelations
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.full_evaluation.category_evaluation.v_names_dates_etc import NamesDatesEtc
from evaluation.full_evaluation.category_evaluation.vi_entity_classification_and_linking import \
    EntityClassificationAndLinking
from evaluation.full_evaluation.category_evaluation.vii_lexical_disambiguation import LexicalDisambiguation
from evaluation.full_evaluation.category_evaluation.viii_attachments import Attachments
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval
from evaluation.single_eval import num_to_score

#  Category names are the same as in the paper (tables 3-5), but all lowercase, and with all punctuation, brackets etc.
#  removed. Except for '+', which is replaced by 'plus', and whitespace ' ' which is replaced by '_'.
#  Sanity checks include the name of the category they are checking, such as multiple_adjectives_sanity_check.
#  Finally, in CP recursion names, "relative clause" is always abbreviated as "rc".
# category_name_to_set_class_and_eval_function = {
#     "pragmatic_coreference_testset": (PragmaticReentrancies, PragmaticReentrancies.compute_testset_results),
#     "pragmatic_coreference_winograd": (PragmaticReentrancies, PragmaticReentrancies.compute_winograd_results),
#     "syntactic_gap_reentrancies": (UnambiguousReentrancies, UnambiguousReentrancies.compute_syntactic_gap_results),
#     "unambiguous_coreference": (UnambiguousReentrancies, UnambiguousReentrancies.compute_unambiguous_coreference_results),
#     "nested_control_and_coordination": (StructuralGeneralization, StructuralGeneralization.computed_nested_control_and_coordination_results),
#     "nested_control_and_coordination_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_nested_control_and_coordination_sanity_check_results),
#     "multiple_adjectives": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_results),
#     "multiple_adjectives_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_sanity_check_results),
#     "centre_embedding": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_results),
#     "centre_embedding_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_sanity_check_results),
#     "cp_recursion": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_results),
#     "cp_recursion_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_sanity_check_results),
#     "cp_recursion_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_results),
#     "cp_recursion_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_sanity_check_results),
#     "cp_recursion_plus_rc": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_results),
#     "cp_recursion_plus_rc_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_sanity_check_results),
#     "cp_recursion_plus_rc_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_results),
#     "cp_recursion_plus_rc_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_sanity_check_results),
#     "long_lists": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_results),
#     "long_lists_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_sanity_check_results),
#     "rare_node_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_node_label_results),
#     "unseen_node_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_node_label_results),
#     "rare_predicate_senses_excl_01": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_sense_results),
#     "unseen_predicate_senses_excl_01": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_sense_results),
#     "rare_edge_labels_ARG2plus": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_edge_label_results),
#     "unseen_edge_labels_ARG2plus": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_edge_label_results),
#     "seen_names": (NamesDatesEtc, NamesDatesEtc.compute_seen_names_results),
#     "unseen_names": (NamesDatesEtc, NamesDatesEtc.compute_unseen_names_results),
#     "seen_dates": (NamesDatesEtc, NamesDatesEtc.compute_seen_dates_results),
#     "unseen_dates": (NamesDatesEtc, NamesDatesEtc.compute_unseen_dates_results),
#     "other_seen_entities": (NamesDatesEtc, NamesDatesEtc.compute_seen_special_entities_results),
#     "other_unseen_entities": (NamesDatesEtc, NamesDatesEtc.compute_unseen_special_entities_results),
#     "types_of_seen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_ne_types_results),
#     "types_of_unseen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_unseen_ne_types_results),
#     "seen_andor_easy_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_andor_easy_wiki_results),
#     "hard_unseen_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_hard_wiki_results),
#     "frequent_predicate_senses_incl_01": (LexicalDisambiguation, LexicalDisambiguation.compute_common_senses_results),
#     "word_ambiguities_handcrafted": (LexicalDisambiguation, LexicalDisambiguation.compute_grapes_word_disambiguation_results),
#     "word_ambiguities_karidi_et_al_2021": (LexicalDisambiguation, LexicalDisambiguation.compute_berts_mouth_results),
#     "pp_attachment": (Attachments, Attachments.compute_pp_results),
#     "unbounded_dependencies": (Attachments, Attachments.compute_unbounded_results),
#     "passives": (Attachments, Attachments.compute_passive_results),
#     "unaccusatives": (Attachments, Attachments.compute_unaccusative_results),
#     "ellipsis": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_ellipsis_results),
#     "multinode_word_meanings": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_multinode_constants_results),
#     "imperatives": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_imperative_results)
# }

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
                                                                tsv="entrancies_unambiguous_coreference_filtered.tsv",
                                                                parent_column=4,
                                                                parent_edge_column=5)),
    # "nested_control_and_coordination": (StructuralGeneralization, StructuralGeneralization.computed_nested_control_and_coordination_results),
    # "nested_control_and_coordination_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_nested_control_and_coordination_sanity_check_results),
    # "multiple_adjectives": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_results),
    # "multiple_adjectives_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_sanity_check_results),
    # "centre_embedding": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_results),
    # "centre_embedding_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_sanity_check_results),
    # "cp_recursion": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_results),
    # "cp_recursion_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_sanity_check_results),
    # "cp_recursion_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_results),
    # "cp_recursion_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_sanity_check_results),
    # "cp_recursion_plus_rc": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_results),
    # "cp_recursion_plus_rc_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_sanity_check_results),
    # "cp_recursion_plus_rc_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_results),
    # "cp_recursion_plus_rc_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_sanity_check_results),
    # "long_lists": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_results),
    # "long_lists_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_sanity_check_results),
    "rare_node_labels": (NodeRecall, SubcategoryMetadata(display_name="Rare node labels",
                                                         tsv="rare_node_labels_test.tsv",
                                                         use_sense=True,
                                                         run_prerequisites=False)),
    "unseen_node_labels": (NodeRecall, SubcategoryMetadata(display_name="Unseen node labels",
                                                           tsv="unseen_node_labels_test_filtered.tsv",
                                                           use_sense=True,
                                                           run_prerequisites=False)),
    "rare_predicate_senses_excl_01": (NodeRecall, SubcategoryMetadata(
        display_name="Rare predicate senses (excl.~\\nl{-01})",
        tsv="rare_senses_filtered.tsv",
        use_sense=True)),
    # "unseen_predicate_senses_excl_01": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_sense_results),
    "rare_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(display_name="Rare edge labels (\\nl{ARG2}+)",
                                                                  tsv="rare_roles_arg2plus_filtered.tsv",
                                                                  use_sense=True
                                  )),
    "unseen_edge_labels_ARG2plus": (EdgeRecall, SubcategoryMetadata(
        display_name="Unseen edge labels (\\nl{ARG2}+)",
        tsv="unseen_roles_new_sentences.tsv", use_sense=True,
        subcorpus_filename="unseen_roles_new_sentences"
    )),
    # "seen_names": (NamesDatesEtc, NamesDatesEtc.compute_seen_names_results),
    # "unseen_names": (NamesDatesEtc, NamesDatesEtc.compute_unseen_names_results),
    # "seen_dates": (NamesDatesEtc, NamesDatesEtc.compute_seen_dates_results),
    # "unseen_dates": (NamesDatesEtc, NamesDatesEtc.compute_unseen_dates_results),
    # "other_seen_entities": (NamesDatesEtc, NamesDatesEtc.compute_seen_special_entities_results),
    # "other_unseen_entities": (NamesDatesEtc, NamesDatesEtc.compute_unseen_special_entities_results),
    # "types_of_seen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_ne_types_results),
    # "types_of_unseen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_unseen_ne_types_results),
    # "seen_andor_easy_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_andor_easy_wiki_results),
    # "hard_unseen_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_hard_wiki_results),
    # "frequent_predicate_senses_incl_01": (LexicalDisambiguation, LexicalDisambiguation.compute_common_senses_results),
    # "word_ambiguities_handcrafted": (LexicalDisambiguation, LexicalDisambiguation.compute_grapes_word_disambiguation_results),
    # "word_ambiguities_karidi_et_al_2021": (LexicalDisambiguation, LexicalDisambiguation.compute_berts_mouth_results),
    # "pp_attachment": (Attachments, Attachments.compute_pp_results),
    # "unbounded_dependencies": (Attachments, Attachments.compute_unbounded_results),
    # "passives": (Attachments, Attachments.compute_passive_results),
    # "unaccusatives": (Attachments, Attachments.compute_unaccusative_results),
    # "ellipsis": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_ellipsis_results),
    # "multinode_word_meanings": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_multinode_constants_results),
    # "imperatives": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_imperative_results)
}





def get_formatted_category_names():
    return "\n".join(category_name_to_set_class_and_metadata.keys())  # TODO linebreak doesn't seem to work in help


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single category.")
    parser.add_argument('-c', '--category_name', type=str, help='Category to evaluate. Possible values are: '
                                                                + get_formatted_category_names())
    parser.add_argument('-g', '--gold_amr_file', type=str, help='Path to gold AMR file. Optional if a GrAPES-specific AMR file', default=None)
    parser.add_argument('-p', '--predicted_amr_file', type=str, help='Path to predicted AMR file. Must contain AMRs '
                                                                     'for all sentences in the gold file, in the same '
                                                                     'order.')
    args = parser.parse_args()
    return args


def get_results(gold_graphs, predicted_graphs, category_name):
    eval_class, info = category_name_to_set_class_and_metadata[category_name]
    # set_class, eval_function = category_name_to_set_class_and_eval_function[category_name]
    set = eval_class(gold_graphs, predicted_graphs, None, "./")
    print("Using", set.__class__.__name__)
    return set.run_single_evaluation(info)


def main():
    args = parse_args()
    eval_class, info = category_name_to_set_class_and_metadata[args.category_name]
    if info.subcorpus_filename is not None:
        gold_graph_path = f"corpus/subcorpora/{info.subcorpus_filename}.txt"
    elif args.gold_amr_file is None:
        print("Please give the path the gold AMR testset file")
        exit(1)
    else:
        gold_graph_path = args.gold_amr_file
    gold_graphs = load(gold_graph_path)
    predicted_graphs = load(args.predicted_amr_file)
    if len(gold_graphs) != len(predicted_graphs):
        raise ValueError("Gold and predicted AMR files must contain the same number of AMRs."
                         "Got " + str(len(gold_graphs)) + " gold AMRs and " + str(len(predicted_graphs))
                         + " predicted AMRs.")
    results = get_results(gold_graphs, predicted_graphs, args.category_name)
    print("Results on " + category_name_to_set_class_and_metadata[args.category_name][1].display_name)
    for row in results:
        metric_name = row[1]
        metric_type = row[2]
        if metric_type == EVAL_TYPE_SUCCESS_RATE:
            wilson_ci = wilson_score_interval(row[3], row[4])
            if row[4] > 0:
                print(f"{metric_name}: {num_to_score(row[3] / row[4])} with Wilson confidence interval "
                      f"[{num_to_score(wilson_ci[0])}, {num_to_score(wilson_ci[1])}] and sample size {row[4]}])")
            else:
                print("ERROR: Division by zero! This means something unexpected went wrong (feel free to contact the "
                      "developers of GrAPES for help, e.g. by filing an issue on GitHub).")
                print(row)
        elif metric_type == EVAL_TYPE_F1:
            print(f"{metric_name}: {num_to_score(row[3])}")
        else:
            print("ERROR: Unexpected evaluation type! This means something unexpected went wrong (feel free to "
                  "contact the developers of GrAPES for help, e.g. by filing an issue on GitHub).")
            print(row)


if __name__ == "__main__":
    main()
