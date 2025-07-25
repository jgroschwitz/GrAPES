from evaluation.file_utils import get_2_columns_from_tsv_by_id, get_graphid2labels_from_tsv_file
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, PREREQS
from evaluation.util import get_node_by_name, get_name


class NETypeRecall(CategoryEvaluation):
    """Identifying named entity types"""

    def read_tsv(self):
        return get_2_columns_from_tsv_by_id(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                            id_column=self.category_metadata.graph_id_column,
                                            column_1=self.category_metadata.label_column
                                            )

    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):
        found = False
        prereq_success = False
        for edge in predicted_amr.edges(role=":name"):  #.edges(role=":name"):
            entity_label = get_node_by_name(edge.source, predicted_amr).target
            name_string = get_name(edge.target, predicted_amr)
            if name_string == target[1]:
                self.add_success(gold_amr, predicted_amr, PREREQS)
                prereq_success = True
                if entity_label == target[0]:
                    self.add_success(gold_amr, predicted_amr)
                    found = True
                break
        if not found:
            self.add_fail(gold_amr, predicted_amr)
        if not prereq_success:
            self.add_fail(gold_amr, predicted_amr, PREREQS)


class NERecall(CategoryEvaluation):
    """Correctly creating attributes for named entities, such as the components of a name"""

    def read_tsv(self):
        return get_graphid2labels_from_tsv_file(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                         graph_id_column=self.category_metadata.graph_id_column,
                                         label_column=self.category_metadata.label_column)

    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):
        found = False
        entity_type = self.category_metadata.subtype
        if entity_type == "other":
            # if not name or date, try both attributes and instances
            gold_value_string = normalize_special_entity_value(target)
            for instance_or_attribute in predicted_amr.instances() + predicted_amr.attributes():
                # and we only need to normalise the one string
                if normalize_special_entity_value(instance_or_attribute.target) == gold_value_string:
                    self.add_success(gold_amr, predicted_amr)
                    found = True
                    break
        else:
            for instance in predicted_amr.instances():
                if instance.target == entity_type:

                    if entity_type == "date-entity":
                        name_string = get_date_string_for_date_instance(predicted_amr, instance)
                    elif entity_type == "name":
                        name_string = get_name_string_for_name_instance(predicted_amr, instance)
                    else:
                        raise ValueError(f"Unknown entity type {entity_type}")
                    # get all the relevant attributes and put them into a string of the same format as the TSV
                    if name_string == target:
                        self.add_success(gold_amr, predicted_amr)
                        found = True
                        break
        if not found:
            self.add_fail(gold_amr, predicted_amr)

def get_name_string_for_name_instance(graph, instance):
    """
    given a graph and an instance in it, returns the " ".join of its sorted target labels
    used only when instance is a name, so the targets are op_i. Sorting them gives the parts of the name in order.
    :param graph:
    :param instance:
    :return: str: e.g. "Capitol Hill"
    """
    name_dict = []
    for attribute in graph.attributes(source=instance.source):
        name_dict.append((attribute.role, attribute.target))
    name_dict.sort()  # in opi order
    name_string = " ".join([t[1] for t in name_dict]).replace("\"", "")
    return name_string


def get_date_string_for_date_instance(graph, instance):
    date_dict = []
    for attribute in graph.attributes(source=instance.source):
        date_dict.append((attribute.role, attribute.target))
    date_dict.sort()
    date_string = " ".join([f"{t[0]} {t[1]}" for t in date_dict])
    return date_string

def normalize_special_entity_value(string):
    return string.replace("\"", "").lower()