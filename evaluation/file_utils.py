import csv
import os

import penman
from penman import load

from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata


def load_corpus_from_folder(folder_path: str):
    """
    :return: list of penman graph objects
    """
    corpus = []
    for file in sorted(os.listdir(folder_path)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            corpus.extend(load(folder_path + filename, encoding="utf8"))
    return corpus


def read_tsv_with_comments(file):
    """
    Gives all rows in a csv file, in the same format as csv.reader would. Except that it excludes all rows that start
    with a # (comment)
    :param file: actual file object (e.g. created via 'with open(path) as file:')
    :return:
    """
    # c.f. https://stackoverflow.com/questions/14158868/python-skip-comment-lines-marked-with-in-csv-dictreader
    reader = csv.reader(filter(lambda row: row[0] != '#', file), delimiter='\t', quotechar=None)
    return reader


def read_label_tsv(root_dir, tsv_file_name, columns=None, graph_id_column=0):
    """
    Reads in labels from columns (default just column 1)
    :return: dict id (str) : labels (str list) of all labels associated with that ID
    """
    if columns is None:
        columns = [1]
    id2labels = dict()
    with open(f"{root_dir}/corpus/{tsv_file_name}", "r", encoding="utf8") as f:
        csvreader = read_tsv_with_comments(f)
        for row in csvreader:
            graph_id = row[graph_id_column]
            labels_here = id2labels.setdefault(graph_id, [])
            if len(columns) == 1:
                # no nested lists if we only want one thing
                label = row[columns[0]]
                labels_here.append(label)
            else:
                # nested lists of things from each column
                by_column = []
                for column in columns:
                    print("getting column", column)
                    label = row[column]
                    by_column.append(label)
                labels_here.append(by_column)
    return id2labels


def read_edge_tsv(root_dir, subcategory_info: SubcategoryMetadata):
    """
    Most TSVs are already formatted as in the defaults, but eg for reentrancies we also need the other parent and edge.
    :param root_dir: root directory path
    :param subcategory_info: SubcategoryMetadata that includes the following:
        first_row_is_header: if true, the first row in the file will be skipped
        graph_id_column: default 0
        source_column: default 1
        edge_column: default 2
        target_column: default 3
        parent_column: default None (for additional parent)
        parent_edge_column: default None (for edge label from additional parent)
    :return: dict id (str) : label list [source_label, edge_label, target_label, (parent_label), (parent_edge_label)]
    """
    id2labels = dict()
    with open(f"{root_dir}/corpus/{subcategory_info.tsv}", "r", encoding="utf8") as f:
        csvreader = read_tsv_with_comments(f)
        is_first_row = True
        for row in csvreader:
            if is_first_row and subcategory_info.first_row_is_header:
                is_first_row = False
                continue
            else:
                is_first_row = False
            graph_id = row[subcategory_info.graph_id_column]
            labels_here = id2labels.setdefault(graph_id, [])
            source_label = row[subcategory_info.source_column]
            edge_label = row[subcategory_info.edge_column]
            target_label = row[subcategory_info.target_column]
            if subcategory_info.parent_column is not None:
                parent_label = row[subcategory_info.parent_column]
                parent_edge_label = row[subcategory_info.parent_edge_column]
                labels_here.append((source_label, edge_label, target_label, parent_label, parent_edge_label))
            else:
                labels_here.append((source_label, edge_label, target_label))
    return id2labels


node_name_alias_counter = 0


def get_graph_for_node_string(node_string: str):
    """
    Reads a "node string" from a tsv file and turns it into a penman graph object. The node_string can have two forms:
    First, an actual penman graph string (may contain more than just one node). In this case, we just decode it.
    Second, a node label (e.g. "person"). In this case, we create a new graph with a single node (with unique name
    that label).
    :param node_string:
    :return:
    """
    global node_name_alias_counter
    is_amr_string = node_string.startswith("(")  # otherwise we just have a node label
    if is_amr_string:
        return penman.decode(node_string)
        # TODO maybe just to be sure, we should replace node names
        #  with globally unique ones
    else:
        node_name_alias_counter += 1
        return penman.decode(f"(x{node_name_alias_counter} / {node_string})")
