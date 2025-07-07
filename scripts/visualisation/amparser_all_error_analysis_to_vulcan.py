import os
from typing import Tuple, Callable

import penman
from amconll import parse_amconll, write_conll

from evaluation.full_evaluation.category_evaluation.category_metadata import category_name_to_set_class_and_metadata, \
    get_categories_by_main_file
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.full_evaluation.evaluation_instance_info import EvaluationInstanceInfo
from scripts.visualisation.amparser_output_to_vulcan import create_pickle_for_error_analysis, \
    create_pickle, make_am_path


def get_evaluator_info_and_ids_for_category(category, gold_amrs, predicted_amrs):

    x: Tuple[Callable, SubcategoryMetadata] = category_name_to_set_class_and_metadata[category]
    eval_class, info = x
    dummy_evaluator = eval_class(gold_amrs, predicted_amrs, info, instance_info)
    ids = dummy_evaluator.get_all_gold_ids()
    return dummy_evaluator, info,  ids


def get_am_sents(info, ids):

    # read in the amconll file(s) of parser predictions
    amconll_sents = []
    files_to_read = [info.subcorpus_filename if info.subcorpus_filename else "testset"]
    if info.extra_subcorpus_filenames is not None:
        files_to_read += info.extra_subcorpus_filenames
    for file in files_to_read:
        try:
            with open(f"{make_am_path(am_path, file)}/AMR-2020_pred.amconll", "r", encoding="utf-8") as f:
                amconll_sents += [s for s in parse_amconll(f, False) if s.attributes["id"] in ids]
        except FileNotFoundError as e:
            if info.name == "pp_attachment":
                pass
            else:
                print("WARNING: could not read in AM conll file for", file, e)
                raise e
    filtered_amconll_path = f"{amconll_outpath}/{info.name}.amconll"
    write_conll(filtered_amconll_path, amconll_sents)
    return amconll_sents


def run_all_for_file(categories, gold_amrs, predicted_amrs):
    for category in categories:

        evaluator, info, ids = get_evaluator_info_and_ids_for_category(category, gold_amrs, predicted_amrs)
        if len(ids) == 0:
            print(f"\n### Skipping {category}: no matching graphs found in corpus\n")
            continue

        amconll_sents = get_am_sents(info, ids)

        # error analysis pickles
        input_error_analysis_pickle_path = f"{root_dir_here}/error_analysis/{instance_info.parser_name}/dictionaries/{info.name}.pickle"
        out_dir = f"{parent}/vulcan_correct_and_incorrect"
        os.makedirs(out_dir, exist_ok=True)
        try:
            create_pickle_for_error_analysis(evaluator, amconll_sents, input_error_analysis_pickle_path, out_dir, verbose=False)
        except FileNotFoundError as e:
            print(f"\n ### Skipping {category}: no error analysis pickle found\n")
            continue

        # by subcorpus
        out_dir = f"{parent}/vulcan_subcorpora"
        pickle_path = f"{out_dir}/{info.name}.pickle"
        os.makedirs(out_dir, exist_ok=True)
        create_pickle(evaluator.gold_amrs, evaluator.predicted_amrs, amconll_sents, pickle_path, verbose=False)


if __name__ == '__main__':
    # just hard-coding everything

    root_dir_here = "../.."
    grapes_categories, testset_categories = get_categories_by_main_file()
    in_path_prefix = "data/processed/parser_outputs/amparser-output"

    # GrAPES

    instance_info = EvaluationInstanceInfo(
        root_dir=root_dir_here,
        path_to_grapes_predictions_file_from_root=f"{in_path_prefix}/full_corpus.txt",
        path_to_full_testset_predictions_file_from_root=f"{in_path_prefix}/testset.txt",
        parser_name="amparser",
    )

    am_path = f"{root_dir_here}/{in_path_prefix}/all_files"

    parent = f"{root_dir_here}/error_analysis/{instance_info.parser_name}/am_trees"
    amconll_outpath = f"{parent}/filtered_amconll_files"
    os.makedirs(amconll_outpath, exist_ok=True)

    print("GrAPES")
    gold_amrs = penman.load(instance_info.gold_grapes_path())
    predicted_amrs = penman.load(instance_info.pred_grapes_file_path())
    run_all_for_file(grapes_categories, gold_amrs, predicted_amrs)

    print("\nAMR 3.0 testset")
    gold_amrs = penman.load(instance_info.gold_testset_path())
    predicted_amrs = penman.load(instance_info.testset_pred_file_path())
    run_all_for_file(testset_categories, gold_amrs, predicted_amrs)




