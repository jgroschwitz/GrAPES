from evaluation.corpus_metrics import _label_exists_in_predicted_labels
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation
from evaluation.util import strip_sense


class NodeRecall(CategoryEvaluation):

    def _get_predicted_labels_based_on_evaluation_case(self, predicted_amr, use_sense=None):
        """
        Get the instances or attributes in the given predicted AMR
        Note that if use_attributes and use_sense are both true, we get the attributes, not the senses.
            If they are both false, we get the instances without their senses.
        :param predicted_amr: AMR to search through
        :param use_sense: if True, get all instances with their senses; otherwise all instances without their senses
        :return: list of either attributes or senses (not both)
        """
        if use_sense is None:
            use_sense = self.category_metadata.use_sense
        if self.category_metadata.use_attributes:
            if self.category_metadata.attribute_label:
                predicted_labels = [attr.target.replace("\"", "") for attr in
                                    predicted_amr.attributes(role=self.category_metadata.attribute_label)]
            else:
                predicted_labels = [attr.target.replace("\"", "") for attr in predicted_amr.attributes()]
        elif use_sense:
            predicted_labels = [instance.target for instance in predicted_amr.instances()]
        else:
            predicted_labels = [strip_sense(instance.target) for instance in predicted_amr.instances()]
        return predicted_labels

    def update_error_analysis(self, graph_id, predicted,
                              target):
        predicted_labels, predicted_labels_no_sense = predicted

        # we only check senses if use_senses=True
        check_senses = predicted_labels is not None

        # we also check without senses if uses_senses=False or we're running prereqs
        if predicted_labels_no_sense is not None:
            label_found = self.find_label(predicted_labels_no_sense, target, False)
            # store the result. If we running prereqs, this sense-less version is the prereqs, otherwise it's main
            # (there's no other way of doing prereqs in NodeRecall)
            error_status = "correct" if label_found else "incorrect"
            error_version = "prereqs" if self.category_metadata.run_prerequisites else "ids"
            self.error_analysis_dict[f"{error_status}_{error_version}"].append(graph_id)
            if not label_found and check_senses:
                # if that failed no need to check with senses
                self.add_fail(graph_id)
                check_senses = False
        # if the prereqs worked and , now check for the full label if use_sense=True
        if check_senses:
            label_found = self.find_label(predicted_labels, target, True)
            error_status = "correct" if label_found else "incorrect"
            self.error_analysis_dict[f"{error_status}_ids"].append(graph_id)

    def get_predictions_for_comparison(self, predicted_amr):
        """
        Extract the relevant labels
        Args:
            predicted_amr: penman Graph
        Returns: relevant node labels, with and without senses. If either isn't needed, it's None instead.
        """
        if not self.category_metadata.use_sense or self.category_metadata.run_prerequisites:
            predicted_labels_no_sense = self._get_predicted_labels_based_on_evaluation_case(
                predicted_amr,
                use_sense=False)
        else:
            predicted_labels_no_sense = None
        # We always need them with senses if use_sense=True
        if self.category_metadata.use_sense:
            predicted_labels = self._get_predicted_labels_based_on_evaluation_case(
                predicted_amr,
                use_sense=True)
        else:
            predicted_labels = None
        return predicted_labels, predicted_labels_no_sense

    def find_label(self, predicted_labels, target_label, use_sense):
        label_found = _label_exists_in_predicted_labels(predicted_labels, target_label, use_sense)
        if not label_found and " " in target_label:
            for target_label_variant in target_label.split(" "):
                if _label_exists_in_predicted_labels(predicted_labels, target_label_variant, use_sense):
                    label_found = True
                    break
        return label_found
