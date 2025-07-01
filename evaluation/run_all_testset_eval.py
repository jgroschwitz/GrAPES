from evaluation.testset.reentrancies import evaluate_syntactic_gap_reentrancies_test,\
    evaluate_pragmatic_reentrancies_test, evaluate_unambiguous_coreference_reentrancies_test
from evaluation.testset.word_ambiguities import evaluate_word_ambiguities_test
from testset.wiki_test import evaluate_hard_wiki_test, evaluate_seen_andor_easy_wiki_test
from testset.rare_senses import evaluate_rare_senses_test
from testset.ne_types import evaluate_ne_types_test
from testset.unseen_and_rare_labels import evaluate_unseen_labels_test, evaluate_rare_labels_test
from testset.unseen_and_rare_roles import evaluate_rare_roles, evaluate_unseen_roles
from testset.special_entities import evaluate_special_entities
from testset.ellipsis import evaluate_ellipsis
import prettytable
from util import num_to_score_with_preceding_0
from evaluation.corpus_metrics import calculate_node_label_recall, calculate_edge_recall_for_tsv_file
from evaluation.file_utils import load_corpus_from_folder
from penman import load


def main():
    # parser_names = ["amparser", "amrbart"]
    parser_names = ["amrbart"]
    field_names = [" "]
    field_names.extend(parser_names)
    table = prettytable.PrettyTable(field_names=field_names)
    table.align = "l"

    root_dir = "../"
    parser_outputs = [load(f"{root_dir}/{parser_name}-output/testset.txt") for parser_name in parser_names]
    gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["1. Coreference"] + ["" for _ in parser_names])
    prag_reent_results = [evaluate_pragmatic_reentrancies_test(gold_amrs=gold_amrs,
                                                                                                  predicted_amrs=parser_output,
                                                                                                  root_dir=root_dir)
                                                    for parser_output in parser_outputs]

    table.add_row(["Prag reent testset (recall)"] + [num_to_score_with_preceding_0(result[1]) for result in prag_reent_results])
    table.add_row(["Prag reent testset (prerequisites)"] + [num_to_score_with_preceding_0(result[0]) for result in prag_reent_results])

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["2. Syntactic reentrancies"] + ["" for _ in parser_names])
    synt_reent_results = [evaluate_syntactic_gap_reentrancies_test(gold_amrs=gold_amrs,
                                                                   predicted_amrs=parser_output,
                                                                   root_dir=root_dir)
                          for parser_output in parser_outputs]
    table.add_row(["Synt gap reent testset (recall)"] + [num_to_score_with_preceding_0(result[1]) for result in synt_reent_results])
    table.add_row(["Synt gap reent testset (prerequisites)"] + [num_to_score_with_preceding_0(result[0]) for result in synt_reent_results])

    unambiguous_coref_results = [evaluate_unambiguous_coreference_reentrancies_test(gold_amrs=gold_amrs,
                                                                   predicted_amrs=parser_output,
                                                                   root_dir=root_dir)
                          for parser_output in parser_outputs]
    table.add_row(["Unambiguous coref testset (recall)"] + [num_to_score_with_preceding_0(result[1]) for result in unambiguous_coref_results])
    table.add_row(["Unambiguous coref testset (prerequisites)"] + [num_to_score_with_preceding_0(result[0]) for result in unambiguous_coref_results])

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["3. Structural Generalization"] + ["" for _ in parser_names])

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["4. Rare and unseen words/senses/roles"] + ["" for _ in parser_names])
    table.add_row(["Rare labels (recall)"] + [num_to_score_with_preceding_0(evaluate_rare_labels_test(gold_amrs=gold_amrs,
                                                                                                      predicted_amrs=parser_output,
                                                                                                      root_dir=root_dir))
                                              for parser_output in parser_outputs])
    table.add_row(["Unseen labels (recall)"] + [num_to_score_with_preceding_0(evaluate_unseen_labels_test(gold_amrs=gold_amrs,
                                                                                                          predicted_amrs=parser_output,
                                                                                                          root_dir=root_dir))
                                                for parser_output in parser_outputs])

    rare_senses_results = [evaluate_rare_senses_test(gold_amrs=gold_amrs, predicted_amrs=parser_output,
                                                     root_dir=root_dir)
                           for parser_output in parser_outputs]
    table.add_row(["Rare senses (excl -01)"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[0]) for result in rare_senses_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[1]) for result in rare_senses_results])

    table.add_row(["Unseen senses (incl -01)"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(calculate_node_label_recall(tsv_file_name="unseen_senses_test.tsv",
                                                                                                     root_dir=root_dir,
                                                                                                     gold_amrs=gold_amrs,
                                                                                                     predicted_amrs=parser_output,
                                                                                                     use_sense=False))
                                           for parser_output in parser_outputs])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(calculate_node_label_recall(tsv_file_name="unseen_senses_test.tsv",
                                                                                              root_dir=root_dir,
                                                                                              gold_amrs=gold_amrs,
                                                                                              predicted_amrs=parser_output,
                                                                                              use_sense=True))
                                    for parser_output in parser_outputs])

    rare_role_results = [evaluate_rare_roles(gold_amrs=gold_amrs, predicted_amrs=parser_output,
                                             root_dir=root_dir)
                         for parser_output in parser_outputs]
    table.add_row(["Rare roles (probably drop)"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[0]) for result in rare_role_results])
    table.add_row(["  - Edge existence"] + [num_to_score_with_preceding_0(result[1]) for result in rare_role_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[2]) for result in rare_role_results])
    table.add_row(["Rare roles ARG2+"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[3]) for result in rare_role_results])
    table.add_row(["  - Edge existence"] + [num_to_score_with_preceding_0(result[4]) for result in rare_role_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[5]) for result in rare_role_results])

    unseen_role_results = [evaluate_unseen_roles(gold_amrs=gold_amrs, predicted_amrs=parser_output,
                                                 root_dir=root_dir)
                           for parser_output in parser_outputs]
    table.add_row(["Unseen roles (probably drop)"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[0]) for result in unseen_role_results])
    table.add_row(["  - Edge existence"] + [num_to_score_with_preceding_0(result[1]) for result in unseen_role_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[2]) for result in unseen_role_results])
    table.add_row(["Unseen roles ARG2+"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[3]) for result in unseen_role_results])
    table.add_row(["  - Edge existence"] + [num_to_score_with_preceding_0(result[4]) for result in unseen_role_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[5]) for result in unseen_role_results])

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["5. Entities"] + ["" for _ in parser_names])
    special_entities_results = [evaluate_special_entities(gold_amrs=gold_amrs, predicted_amrs=parser_output,
                                                          root_dir=root_dir)
                                for parser_output in parser_outputs]
    table.add_row(["Special entities"] + ["" for _ in parser_names])
    table.add_row(["  - Names (seen)"] + [num_to_score_with_preceding_0(result[5]) for result in special_entities_results])
    table.add_row(["  - Names (unseen)"] + [num_to_score_with_preceding_0(result[2]) for result in special_entities_results])
    table.add_row(["  - Dates (seen)"] + [num_to_score_with_preceding_0(result[4]) for result in special_entities_results])
    table.add_row(["  - Dates (unseen)"] + [num_to_score_with_preceding_0(result[1]) for result in special_entities_results])
    table.add_row(["  - Other (seen)"] + [num_to_score_with_preceding_0(result[3]) for result in special_entities_results])
    table.add_row(["  - Other (unseen)"] + [num_to_score_with_preceding_0(result[0]) for result in special_entities_results])

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["6. NE classification and linking"] + ["" for _ in parser_names])
    ne_type_results = [evaluate_ne_types_test(gold_amrs=gold_amrs, predicted_amrs=parser_output, root_dir=root_dir)
                       for parser_output in parser_outputs]
    table.add_row(["NE types (seen)"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[2]) for result in ne_type_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[3]) for result in ne_type_results])
    table.add_row(["NE types (unseen)"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[0]) for result in ne_type_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[1]) for result in ne_type_results])

    table.add_row(["Seen and/or easy Wiki (recall)"]
                  + [num_to_score_with_preceding_0(evaluate_seen_andor_easy_wiki_test(gold_amrs=gold_amrs, predicted_amrs=parser_output,
                                                                                      root_dir=root_dir))
                     for parser_output in parser_outputs])
    table.add_row(["Hard Wiki (recall)"]
                  + [num_to_score_with_preceding_0(evaluate_hard_wiki_test(gold_amrs=gold_amrs, predicted_amrs=parser_output,
                                                                           root_dir=root_dir))
                     for parser_output in parser_outputs])

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["7. Sense disambiguation"] + ["" for _ in parser_names])
    table.add_row(["(rare senses in section 4)"] + ["" for _ in parser_names])
    table.add_row(["Common roles"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(calculate_node_label_recall(tsv_file_name="common_senses_labeled.tsv",
                                                                                                     root_dir=root_dir,
                                                                                                     gold_amrs=gold_amrs,
                                                                                                     predicted_amrs=parser_output,
                                                                                                     use_sense=False))
                                           for parser_output in parser_outputs])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(calculate_node_label_recall(tsv_file_name="common_senses_labeled.tsv",
                                                                                              root_dir=root_dir,
                                                                                              gold_amrs=gold_amrs,
                                                                                              predicted_amrs=parser_output,
                                                                                              use_sense=True))
                                    for parser_output in parser_outputs])

    word_ambiguities_results = [evaluate_word_ambiguities_test(gold_amrs=gold_amrs, predicted_amrs=parser_output,
                                root_dir=root_dir) for parser_output in parser_outputs]
    table.add_row(["Word ambiguities"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[0]) for result in word_ambiguities_results])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[1]) for result in word_ambiguities_results])

    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["8. Attachments"] + ["" for _ in parser_names])

    unaccusative_scores = [calculate_edge_recall_for_tsv_file(tsv_file_name="unaccusatives2_filtered.tsv",
                                                             gold_amrs=gold_amrs, predicted_amrs=parser_out,
                                                              root_dir=root_dir)
                                                              for parser_out in parser_outputs]
    table.add_row(["Unaccusatives"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[0]) for result in unaccusative_scores])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[1]) for result in unaccusative_scores])
    passive_scores = [calculate_edge_recall_for_tsv_file(tsv_file_name="passives_filtered.tsv",
                                                              gold_amrs=gold_amrs, predicted_amrs=parser_out,
                                                              root_dir=root_dir)
                           for parser_out in parser_outputs]
    table.add_row(["Passives"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisites"] + [num_to_score_with_preceding_0(result[0]) for result in passive_scores])
    table.add_row(["  - Recall"] + [num_to_score_with_preceding_0(result[1]) for result in passive_scores])


    table.add_row([""] + ["" for _ in parser_names])
    table.add_row(["9. Unusual word-node relations"] + ["" for _ in parser_names])
    ellipsis_results = [evaluate_ellipsis(gold_amrs=gold_amrs, predicted_amrs=parser_output, root_dir=root_dir)
                        for parser_output in parser_outputs]
    table.add_row(["Ellipsis"] + ["" for _ in parser_names])
    table.add_row(["  - Prerequisite (have 1+)"] + [num_to_score_with_preceding_0(result[0]) for result in ellipsis_results])
    table.add_row(["  - Recall (have 2+)"] + [num_to_score_with_preceding_0(result[1]) for result in ellipsis_results])

    print(table)


if __name__ == '__main__':
    main()
