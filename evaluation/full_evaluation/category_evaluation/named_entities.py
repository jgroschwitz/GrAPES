from amrbank_analysis.get_unseen_names_and_dates import get_date_string_for_date_instance, \
    get_name_string_for_name_instance
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, PREREQS
from evaluation.testset.ne_types import get_2_columns_from_tsv_by_id
from evaluation.testset.special_entities import get_graphid2labels_from_tsv_file, normalize_special_entity_value
from evaluation.util import get_node_by_name, get_name


class NETypeRecall(CategoryEvaluation):
    """Identifying named entity types"""

    def read_tsv(self):
        return get_2_columns_from_tsv_by_id(f"{self.corpus_path}/{self.category_metadata.tsv}",
                                            id_column=self.category_metadata.graph_id_column,
                                            column_2=self.category_metadata.label_column
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
