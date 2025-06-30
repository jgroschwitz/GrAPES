from typing import List

from evaluation.corpus_metrics import calculate_node_label_recall
from evaluation.file_utils import load_corpus_from_folder
from penman import load, Graph, encode
import csv
from amrbank_analysis.get_unseen_names_and_dates import get_name_string_for_name_instance, \
    get_date_string_for_date_instance


def evaluate_special_entities(gold_amrs=None, predicted_amrs=None, parser_name=None, root_dir="../../"):
    """
    gets recall on all special entities: names, dates, and others; seen and unseen
    names: alphanumerically-sorted attributes of a "name" instance
    dates: alphanumerically-sorted attributes of a "date-entity" instance
    others: instances (not attributes), including senses
    :return: 6-tuple of: unseen_special_entities_recall, unseen_date_recall, unseen_name_recall,
           seen_special_entities_recall, seen_date_recall, seen_name_recall
    """
    assert predicted_amrs is not None or parser_name is not None

    # other special entities like string-entity and monetary-quantity
    # get matches for instances (but not attributes) including their senses
    unseen_special_entities_recall = calculate_node_label_recall("special_entities_unseen.tsv",
                                                                 gold_amrs=gold_amrs,
                                                                 predicted_amrs=predicted_amrs,
                                                                 parser_name=parser_name,
                                                                 root_dir=root_dir)
    seen_special_entities_recall = calculate_node_label_recall("special_entities_seen.tsv",
                                                               gold_amrs=gold_amrs,
                                                               predicted_amrs=predicted_amrs,
                                                               parser_name=parser_name,
                                                               root_dir=root_dir)

    # get gold AMRs from file if not given
    if gold_amrs is None:
        gold_amrs = load_corpus_from_folder(f"{root_dir}/external_resources/amrs/split/test/")
    # obsolete, I think
    if predicted_amrs is None:
        predicted_amrs = load(f"{root_dir}/{parser_name}-output/testset.txt")

    unseen_date_recall = calculate_date_recall(f"{root_dir}/corpus/unseen_dates.tsv", gold_amrs, predicted_amrs)
    seen_date_recall = calculate_date_recall(f"{root_dir}/corpus/seen_dates.tsv", gold_amrs, predicted_amrs)

    unseen_name_recall = calculate_name_recall(f"{root_dir}/corpus/unseen_names.tsv", gold_amrs, predicted_amrs)
    seen_name_recall = calculate_name_recall(f"{root_dir}/corpus/seen_names.tsv", gold_amrs, predicted_amrs)

    return unseen_special_entities_recall, unseen_date_recall, unseen_name_recall, \
        seen_special_entities_recall, seen_date_recall, seen_name_recall


def calculate_other_special_entity_successes_and_sample_size(tsv_file_path,
                                                             gold_amrs: List[Graph],
                                                             predicted_amrs: List[Graph]):
    id2labels = get_graphid2labels_from_tsv_file(tsv_file_path, label_column=3)
    special_entity_recalled = 0
    special_entity_total = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels:
            special_entity_total += len(id2labels[gold_amr.metadata['id']])
            for gold_value_string in id2labels[gold_amr.metadata['id']]:
                gold_value_string = normalize_special_entity_value(gold_value_string)
                found = False
                for instance in predicted_amr.instances():
                    if normalize_special_entity_value(instance.target) == gold_value_string:
                        found = True
                for attribute in predicted_amr.attributes():
                    if normalize_special_entity_value(attribute.target) == gold_value_string:
                        found = True
                if found:
                    special_entity_recalled += 1
    return special_entity_recalled, special_entity_total


def normalize_special_entity_value(string):
    return string.replace("\"", "").lower()

def calculate_date_recall(tsv_file_path, gold_amrs, predicted_amrs):
    date_recalled, date_total = calculate_date_successes_and_sample_size(tsv_file_path, gold_amrs,
                                                                         predicted_amrs)
    date_recall = date_recalled / date_total if date_total > 0 else 1.0
    return date_recall


def calculate_date_successes_and_sample_size(tsv_file_path, gold_amrs, predicted_amrs):
    id2labels_unseen_dates = get_graphid2labels_from_tsv_file(tsv_file_path)
    date_recalled = 0
    date_total = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels_unseen_dates:
            date_total += len(id2labels_unseen_dates[gold_amr.metadata['id']])
            for gold_name_string in id2labels_unseen_dates[gold_amr.metadata['id']]:
                for instance in predicted_amr.instances():
                    if instance.target == "date-entity":
                        name_string = get_date_string_for_date_instance(predicted_amr, instance)
                        if name_string == gold_name_string:
                            date_recalled += 1
                            break
    return date_recalled, date_total

def calculate_special_entity_successes_and_sample_size(id2labels_entities, gold_amrs, predicted_amrs, entity_type):
    """
    For names, dates, and others such as phone numbers. Finds matches on the actual contents (e.g. the phone number itself)
    Args:
        id2labels_entities: dict from graph ids to labels to match
        gold_amrs: list of Penman graphs
        predicted_amrs: "
        entity_type: str: name, date-entity, or other.
    Returns: found, total
    """
    if entity_type == "date-entity":
        fun = get_date_string_for_date_instance
    elif entity_type == "name":
        fun = get_name_string_for_name_instance

    recalled = 0
    total = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels_entities:
            gold_strings = id2labels_entities[gold_amr.metadata['id']]
            total += len(gold_strings)
            for gold_value_string in gold_strings:
                if entity_type == "other":
                    # if not name or date, try both attributes and instances
                    gold_value_string = normalize_special_entity_value(gold_value_string)
                    for instance_or_attribute in predicted_amr.instances() + predicted_amr.attributes():
                        # and we only need to normalise the one string
                        if normalize_special_entity_value(instance_or_attribute.target) == gold_value_string:
                            recalled += 1
                            break
                else:
                    for instance in predicted_amr.instances():
                        if instance.target == entity_type:
                            # get all the relevant attributes and put them into a string of the same format as the TSV
                            name_string = fun(predicted_amr, instance)
                            if name_string == gold_value_string:
                                recalled += 1
                                break
    return recalled, total


def calculate_name_recall(tsv_file_path, gold_amrs, predicted_amrs):
    name_recalled, name_total = calculate_name_successes_and_sample_size(tsv_file_path, gold_amrs, predicted_amrs)
    name_recall = name_recalled / name_total if name_total > 0 else 1.0
    return name_recall


def calculate_name_successes_and_sample_size(tsv_file_path, gold_amrs, predicted_amrs):
    """
    gold AMRs are only needed to match the IDs in the TSV to the predicted graphs (which don't have usually IDs)
    :param tsv_file_path:
    :param gold_amrs:
    :param predicted_amrs:
    :return: int, int: the number of correct names, the number of total names
    """
    id2labels_unseen_names = get_graphid2labels_from_tsv_file(tsv_file_path)
    name_recalled = 0
    name_total = 0
    for gold_amr, predicted_amr in zip(gold_amrs, predicted_amrs):
        if gold_amr.metadata['id'] in id2labels_unseen_names:
            name_total += len(id2labels_unseen_names[gold_amr.metadata['id']])
            for gold_name_string in id2labels_unseen_names[gold_amr.metadata['id']]:
                for instance in predicted_amr.instances():
                    if instance.target == "name":
                        name_string = get_name_string_for_name_instance(predicted_amr, instance)
                        if name_string == gold_name_string:
                            name_recalled += 1
                            break
    return name_recalled, name_total


def get_graphid2labels_from_tsv_file(filepath, graph_id_column=0, label_column=1):
    id2labels = dict()
    with open(filepath, "r") as f:
        csvreader = csv.reader(f, delimiter='\t', quotechar=None)
        for row in csvreader:
            graph_id = row[graph_id_column]
            label = row[label_column]
            labels_here = id2labels.setdefault(graph_id, [])
            labels_here.append(label)
    return id2labels


def main():
    gold_amrs = load("../../corpus/testset.txt")
    predicted_amrs = load("../../amrbart-output/testset.txt")
    tsv_filename = "../../corpus/unseen_special_entities.tsv"
    print(calculate_other_special_entity_successes_and_sample_size(tsv_filename, gold_amrs, predicted_amrs))


if __name__ == '__main__':
    main()
