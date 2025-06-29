from typing import Dict
import pickle

from evaluation.file_utils import load_corpus_from_folder
from evaluation.util import num_to_score
from evaluation.full_evaluation.wilson_score_interval import wilson_score_interval

from penman import load

from evaluation.full_evaluation.category_evaluation.category_evaluation import EVAL_TYPE_SUCCESS_RATE, EVAL_TYPE_F1
from evaluation.full_evaluation.category_evaluation.i_pragmatic_reentrancies import PragmaticReentrancies
from evaluation.full_evaluation.category_evaluation.ii_unambiguous_reentrancies import UnambiguousReentrancies
from evaluation.full_evaluation.category_evaluation.iii_structural_generalization import StructuralGeneralization
from evaluation.full_evaluation.category_evaluation.iv_rare_unseen_nodes_edges import RareUnseenNodesEdges
from evaluation.full_evaluation.category_evaluation.v_names_dates_etc import NamesDatesEtc
from evaluation.full_evaluation.category_evaluation.vi_entity_classification_and_linking import EntityClassificationAndLinking
from evaluation.full_evaluation.category_evaluation.vii_lexical_disambiguation import LexicalDisambiguation
from evaluation.full_evaluation.category_evaluation.viii_attachments import Attachments
from evaluation.full_evaluation.category_evaluation.ix_nontrivial_word2node_relations import NontrivialWord2NodeRelations
from evaluate_single_category import category_name_to_set_class_and_metadata

root_dir = "../../"
path_to_parser_outputs = f"{root_dir}/data/raw/parser_outputs/"



def load_parser_output(parser_name, subcorpus_name):
    return load(f"{path_to_parser_outputs}/{parser_name}-output/{subcorpus_name}.txt")

# TODO check orders and completeness
bunch2subcategory = {
    "1. Pragmatic Reentrancies": ["pragmatic_coreference_testset", "pragmatic_coreference_winograd"],
    "4. Rare Unseen Nodes Edges": ["rare_node_labels", "unseen_node_labels", "rare_predicate_senses_excl_01",
                         "rare_edge_labels_ARG2plus", "unseen_edge_labels_ARG2plus"],
    "2. Unambiguous Reentrancies": ["syntactic_gap_reentrancies", "unambiguous_coreference"],
    "8. Attachments": ["pp_attachment", "unbounded_dependencies", "passives", "unaccusatives"],
    "6. Entity Classification And Linking": ["seen_andor_easy_wiki_links", "hard_unseen_wiki_links"],
    "5. Names Dates Etc": ["seen_names", "unseen_names", "seen_dates", "unseen_dates", "other_seen_entities",
                           "other_unseen_entities",  "types_of_seen_named_entities", "types_of_unseen_named_entities"]
}


def create_results_pickle():
    # parser_names = ["amparser", "cailam", "amrbart"]
    parser_names = ["amparser"]

    gold_amrs = load(f"{root_dir}/data/raw/gold/test.txt")

    parser_name2rows = dict()

    for parser_name in parser_names:
        testset_parser_outs = load_parser_output(parser_name, subcorpus_name="testset")

        print("RESULTS FOR", parser_name)

        all_result_rows = []
        parser_name2rows[parser_name] = all_result_rows

        for bunch in sorted(bunch2subcategory.keys()):

            all_result_rows.append([bunch])
            print(bunch)

            for subcategory in bunch2subcategory[bunch]:
                eval_class, info = category_name_to_set_class_and_metadata[subcategory]
                if info.subcorpus_filename is None:
                    print("Using test set")
                    predictions = testset_parser_outs
                    golds = gold_amrs
                else:
                    print("Using", info.subcorpus_filename)
                    predictions = load_parser_output(parser_name, subcorpus_name=info.subcorpus_filename)
                    golds = load(f"{root_dir}/corpus/subcorpora/{info.subcorpus_filename}.txt")

                    # Special case for PP attachment: they're in separate files
                    # I don't know why the evaluation was even working because we were only looking at
                    # the files in pp_attachment.txt, which don't actually get evaluated.
                    if info.subcorpus_filename == "pp_attachment":
                        print("Concatenating PP files")
                        for filename in ["see_with", "read_by", "bought_for", "keep_from", "give_up_in"]:
                            more_graphs = load(f"{root_dir}/corpus/subcorpora/{filename}.txt")
                            golds += more_graphs
                            more_graphs = load_parser_output(parser_name, subcorpus_name=filename)
                            predictions += more_graphs
                    assert len(golds) == len(predictions) and len(golds) > 0
                set = eval_class(golds, predictions, parser_name, root_dir)
                rows = set.run_single_evaluation(info)
                all_result_rows += rows


        # category_1_evaluation = PragmaticReentrancies(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_1_evaluation.get_result_rows()
        #
        # all_result_rows.append(["2. Unambiguous Reentrancies"])
        # print("2")
        #
        #
        # category_2_evaluation = UnambiguousReentrancies(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_2_evaluation.get_result_rows()
        #
        # all_result_rows.append(["3. Structural Generalization"])
        # print("3")
        #
        # category_3_evaluation = StructuralGeneralization(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_3_evaluation.get_result_rows()
        #
        # all_result_rows.append(["4. Rare Unseen Nodes Edges"])
        # print("4")
        #
        # subcategories = ["rare_node_labels", "unseen_node_labels", "rare_predicate_senses_excl_01",
        #                  "rare_edge_labels_ARG2plus", "unseen_edge_labels_ARG2plus"]
        # for subcategory in subcategories:
        #     eval_class, info = category_name_to_set_class_and_metadata[subcategory]
        #     if info.subcorpus_filename is None:
        #         predictions = testset_parser_outs
        #         golds = gold_amrs
        #     else:
        #         predictions = load_parser_output(parser_name, subcorpus_name=info.subcorpus_filename)
        #         golds = load(f"{root_dir}/corpus/subcorpora/{info.subcorpus_filename}.txt")
        #     set = eval_class(golds, predictions, parser_name, root_dir)
        #     print("Using", set.__class__.__name__)
        #     rows = set.run_single_evaluation(info)
        #     all_result_rows += rows

        #
        # category_4_evaluation = RareUnseenNodesEdges(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_4_evaluation.get_result_rows()
        #
        # all_result_rows.append(["5. Names Dates Etc"])
        # print("5")
        #
        # category_5_evaluation = NamesDatesEtc(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_5_evaluation.get_result_rows()
        #
        # all_result_rows.append(["6. Entity Classification And Linking"])
        # print("6")
        #
        # category_6_evaluation = EntityClassificationAndLinking(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_6_evaluation.get_result_rows()
        #
        # all_result_rows.append(["7. Lexical Disambiguation"])
        # print("7")
        #
        # category_7_evaluation = LexicalDisambiguation(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_7_evaluation.get_result_rows()
        #
        # all_result_rows.append(["8. Attachments"])
        # print("8")
        #
        # category_8_evaluation = Attachments(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_8_evaluation.get_result_rows()
        #
        # all_result_rows.append(["9. Nontrivial Word2Node Relations"])
        # print("9")
        #
        # category_9_evaluation = NontrivialWord2NodeRelations(gold_amrs, testset_parser_outs, parser_name, root_dir)
        # all_result_rows += category_9_evaluation.get_result_rows()

        print("All result rows")
        print(all_result_rows)
        print_pretty_table(all_result_rows)

    pickle.dump(parser_name2rows, open(f"{root_dir}/data/processed/results/results_table.pickle", "wb"))

def print_pretty_table(result_rows):
    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["Dataset", "Metric", "Score", "Wilson CI", "Sample size"]
    table.align = "l"
    for row in result_rows:
        eval_type = _get_row_evaluation_type(row)
        if eval_type == EVAL_TYPE_SUCCESS_RATE:
            wilson_ci = wilson_score_interval(row[3], row[4])
            if row[4] > 0:
                table.add_row([row[0], row[1], num_to_score(row[3]/row[4]),
                               f"[{num_to_score(wilson_ci[0])}, {num_to_score(wilson_ci[1])}]", row[4]])
            else:
                print("Division by zero!")
                print(row)
        elif eval_type == EVAL_TYPE_F1:
            table.add_row([row[0], row[1],  num_to_score(row[3]), "", ""])
        elif eval_type == 1:
            table.add_row(["", "", "", "", ""])
            table.add_row([row[0], "", "", "", ""])
        else:
            print(row)
            raise Exception(f"Unknown evaluation type: {eval_type}")
    print(table)


def make_latex_table(root_dir: str):
    result_rows_by_parser_name = pickle.load(open(root_dir + "/results_table.pickle", "rb"))
    results_rows_by_column = zip(result_rows_by_parser_name["amparser"],
                                 result_rows_by_parser_name["cailam"],
                                 result_rows_by_parser_name["amrbart"])

    set_to_scores = dict()
    current_scores = None

    with open(root_dir + "/latex_results_table.txt", "w") as f:
        set_id = ""
        shade_row = True
        for parser_rows in results_rows_by_column:
            is_title_row = len(parser_rows[0]) == 1
            if is_title_row:
                set_id = parser_rows[0][0][0]
                current_scores = [[], [], []]
                set_to_scores[parser_rows[0][0]] = current_scores
                continue
            dataset_name = parser_rows[0][0]
            metric_name = parser_rows[0][1]

            is_unlabeled_edge_row = metric_name == "Unlabeled edge recall"
            is_smatch_row = "smatch" in metric_name.lower()
            if is_unlabeled_edge_row or is_smatch_row:
                continue

            is_sanity_check_row = "sanity" in dataset_name.lower()
            is_prereq_row = "prereq" in metric_name.lower()
            if set_id in ["1", "3", "4", "7"]:  # the sets that start a new table
                shade_row = True
            else:
                if is_sanity_check_row or dataset_name.strip() == "":
                    pass
                else:
                    shade_row = not shade_row

            shading_prefix = "\\rowcolor{lightlightlightgray}" if shade_row else ""
            latex_line = f"\t\t{shading_prefix}{set_id} & {dataset_name} & {metric_name}"

            is_success_rate_row = parser_rows[0][2] == EVAL_TYPE_SUCCESS_RATE
            if is_success_rate_row and not "precision" in metric_name.lower():
                assert parser_rows[0][4] == parser_rows[1][4] == parser_rows[2][4]
            for i, row in enumerate(parser_rows):
                if is_success_rate_row:
                    wilson_ci = wilson_score_interval(row[3], row[4])
                    score = num_to_score(row[3] / row[4])
                    if not (is_prereq_row or is_sanity_check_row):
                        current_scores[i].append(row[3] / row[4])
                    if len(score) == 2:
                        score = "\\phantom{1}" + score
                    lower_bound = num_to_score(wilson_ci[0])
                    upper_bound = num_to_score(wilson_ci[1])
                    if len(lower_bound) == 2:
                        lower_bound = lower_bound
                    if len(upper_bound) == 2:
                        upper_bound = "\\phantom{1}" + upper_bound
                    # latex_line += f" & ${score}_{{{lower_bound}}}^{{{upper_bound}}}$"
                    latex_line += f" & \\successScore{{{score}}}{{{lower_bound}}}{{{upper_bound}}}{{\\phantom{{1}}}}"
                else:
                    if not (is_prereq_row or is_sanity_check_row):
                        current_scores[i].append(row[3])
                    latex_line += f" & {num_to_score(row[3])}\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ "

            if is_success_rate_row:
                latex_line += f" & {parser_rows[0][4]}\\\\\n"
            else:
                latex_line += f" & \\\\\n"

            f.write(latex_line)

            set_id = ""

    with open(root_dir + "/latex_compact_table.txt", "w") as f:
        for set_name, scores_by_parser in set_to_scores.items():
            # print(set_name)
            # print(scores_by_parser)
            latex_line = f"\t\t{set_name}"
            for scores in scores_by_parser:
                # average the scores
                score = sum(scores) / len(scores)
                latex_line += f" & {num_to_score(score)}"
            latex_line += "\\\\\n"
            f.write(latex_line)



    # 1 & Ambiguous
    # coreference & Recall & $6
    # _
    # {1} ^ {15}$ & & $39
    # _
    # {26} ^ {47}$ \todo
    # {put in actual
    # numbers} & ?\ \


# TODO: make a table that is average for categories (can't have wilson scores here,
#  because we DON'T want to normalize by sample size here)

# TODO: make a structural generalization table, broken down by sizes.


def _get_row_evaluation_type(row):
    if len(row) >= 3:
        return row[2]
    else:
        return len(row)


def main():
    create_results_pickle()
    # make_latex_table("../../")


if __name__ == '__main__':
    main()