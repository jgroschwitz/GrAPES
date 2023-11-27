import argparse
from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_F1, EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.i_pragmatic_reentrancies import PragmaticReentrancies
from evaluation.full_evaluation.category_evaluation.ii_unambiguous_reentrancies import UnambiguousReentrancies
from evaluation.full_evaluation.category_evaluation.iii_structural_generalization import StructuralGeneralization
from evaluation.full_evaluation.category_evaluation.iv_rare_unseen_nodes_edges import RareUnseenNodesEdges
from evaluation.full_evaluation.category_evaluation.ix_nontrivial_word2node_relations import \
    NontrivialWord2NodeRelations
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
category_name_to_set_class_and_eval_function = {
    "pragmatic_coreference_testset": (PragmaticReentrancies, PragmaticReentrancies.compute_testset_results),
    "pragmatic_coreference_winograd": (PragmaticReentrancies, PragmaticReentrancies.compute_winograd_results),
    "syntactic_gap_reentrancies": (UnambiguousReentrancies, UnambiguousReentrancies.compute_syntactic_gap_results),
    "unambiguous_coreference": (UnambiguousReentrancies, UnambiguousReentrancies.compute_unambiguous_coreference_results),
    "nested_control_and_coordination": (StructuralGeneralization, StructuralGeneralization.computed_nested_control_and_coordination_results),
    "nested_control_and_coordination_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_nested_control_and_coordination_sanity_check_results),
    "multiple_adjectives": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_results),
    "multiple_adjectives_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_multiple_adjectives_sanity_check_results),
    "centre_embedding": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_results),
    "centre_embedding_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_centre_embedding_sanity_check_results),
    "cp_recursion": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_results),
    "cp_recursion_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_sanity_check_results),
    "cp_recursion_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_results),
    "cp_recursion_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_coref_sanity_check_results),
    "cp_recursion_plus_rc": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_results),
    "cp_recursion_plus_rc_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_sanity_check_results),
    "cp_recursion_plus_rc_plus_coreference": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_results),
    "cp_recursion_plus_rc_plus_coreference_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_cp_recursion_with_rc_and_coref_sanity_check_results),
    "long_lists": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_results),
    "long_lists_sanity_check": (StructuralGeneralization, StructuralGeneralization.compute_long_lists_sanity_check_results),
    "rare_node_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_node_label_results),
    "unseen_node_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_node_label_results),
    "rare_predicate_senses_excl_01": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_sense_results),
    "unseen_predicate_senses_excl_01": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_sense_results),
    "rare_edge_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_rare_edge_label_results),
    "unseen_edge_labels": (RareUnseenNodesEdges, RareUnseenNodesEdges.compute_unseen_edge_label_results),
    "seen_names": (NamesDatesEtc, NamesDatesEtc.compute_seen_names_results),
    "unseen_names": (NamesDatesEtc, NamesDatesEtc.compute_unseen_names_results),
    "seen_dates": (NamesDatesEtc, NamesDatesEtc.compute_seen_dates_results),
    "unseen_dates": (NamesDatesEtc, NamesDatesEtc.compute_unseen_dates_results),
    "other_seen_entities": (NamesDatesEtc, NamesDatesEtc.compute_seen_special_entities_results),
    "other_unseen_entities": (NamesDatesEtc, NamesDatesEtc.compute_unseen_special_entities_results),
    "types_of_seen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_ne_types_results),
    "types_of_unseen_named_entities": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_unseen_ne_types_results),
    "seen_andor_easy_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_seen_andor_easy_wiki_results),
    "hard_unseen_wiki_links": (EntityClassificationAndLinking, EntityClassificationAndLinking.compute_hard_wiki_results),
    "frequent_predicate_senses_incl_01": (LexicalDisambiguation, LexicalDisambiguation.compute_common_senses_results),
    "word_ambiguities_handcrafted": (LexicalDisambiguation, LexicalDisambiguation.compute_grapes_word_disambiguation_results),
    "word_ambiguities_karidi_et_al_2021": (LexicalDisambiguation, LexicalDisambiguation.compute_berts_mouth_results),
    "pp_attachment": (Attachments, Attachments.compute_pp_results),
    "unbounded_dependencies": (Attachments, Attachments.compute_unbounded_results),
    "passives": (Attachments, Attachments.compute_passive_results),
    "unaccusatives": (Attachments, Attachments.compute_unaccusative_results),
    "ellipsis": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_ellipsis_results),
    "multinode_word_meanings": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_multinode_constants_results),
    "imperatives": (NontrivialWord2NodeRelations, NontrivialWord2NodeRelations.compute_imperative_results)
}


def get_formatted_category_names():
    return "\n".join(category_name_to_set_class_and_eval_function.keys())  # TODO linebreak doesn't seem to work in help


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single category.")
    parser.add_argument('-c', '--category_name', type=str, help='Category to evaluate. Possible values are: '
                                                                + get_formatted_category_names())
    parser.add_argument('-g', '--gold_amr_file', type=str, help='Path to gold AMR file')
    parser.add_argument('-p', '--predicted_amr_file', type=str, help='Path to predicted AMR file. Must contain AMRs '
                                                                     'for all sentences in the gold file, in the same '
                                                                     'order.')
    args = parser.parse_args()
    return args


def get_results(gold_graphs, predicted_graphs, category_name):
    set_class, eval_function = category_name_to_set_class_and_eval_function[category_name]
    set = set_class(gold_graphs, predicted_graphs, None, "./")
    return eval_function(set, gold_graphs, predicted_graphs)


def main():
    args = parse_args()
    gold_graphs = load(args.gold_amr_file)
    predicted_graphs = load(args.predicted_amr_file)
    if len(gold_graphs) != len(predicted_graphs):
        raise ValueError("Gold and predicted AMR files must contain the same number of AMRs."
                         "Got " + str(len(gold_graphs)) + " gold AMRs and " + str(len(predicted_graphs))
                         + " predicted AMRs.")
    results = get_results(gold_graphs, predicted_graphs, args.category_name)
    print("Results on " + args.category_name)
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
