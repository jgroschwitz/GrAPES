from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation
from evaluation.graph_matcher import equals_modulo_isomorphy
from evaluation.novel_corpus.berts_mouth import IDS_FOR_EXACT_MATCH, EDGE_LABELS_AND_REIFICATIONS, \
    is_relation_present_in_graph
from evaluation.util import get_node_by_name


class WordDisambiguationBertsMouth(CategoryEvaluation):
    def update_results(self, gold_amr, predicted_amr, target=None, predictions_for_comparison=None):

        if gold_amr.metadata["suppl"] in IDS_FOR_EXACT_MATCH:
            if equals_modulo_isomorphy(gold_amr, predicted_amr):
                self.add_success(gold_amr, predicted_amr)
            else:
                self.add_fail(gold_amr, predicted_amr)
        else:
            found_relation = False
            for edge_label, reification in EDGE_LABELS_AND_REIFICATIONS:
                if is_relation_present_in_graph(edge_label, gold_amr, reification):
                    found_relation = True
                    if is_relation_present_in_graph(edge_label, predicted_amr, reification):
                        self.add_success(gold_amr, predicted_amr)
                    else:
                        self.add_fail(gold_amr, predicted_amr)
                    break
            if not found_relation:
                root_label = get_node_by_name(gold_amr.top, gold_amr).target
                if len([inst for inst in predicted_amr.instances() if inst.target == root_label]) >= 1:
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)
