from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.novel_corpus.word_disambiguation import get_target_instances, is_relation_present_in_graph, \
    EDGE_LABELS_AND_REIFICATIONS, evaluate_word_disambiguation
from evaluation.util import filter_amrs_for_name


class WordDisambiguationHandcrafted(CategoryEvaluation):
    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):
        """
        This is the main function for evaluating word disambiguation for the handwritten sentences (and a few
        from the test corpus).
        """
        try:
            label = gold_amr.metadata["label"]
            if " " in label:
                # print("first case")
                edge_label, target_label = gold_amr.metadata["label"].split(" ")
                target_instances = get_target_instances(edge_label, predicted_amr)
                found = False
                for target_instance in target_instances:
                    if target_instance.target == target_label:
                        self.add_success(gold_amr, predicted_amr)
                        found = True
                        break
                if not found:
                    self.add_fail(gold_amr, predicted_amr)

            elif label.startswith(":"):
                # print("second case")
                edge_label = label
                if is_relation_present_in_graph(edge_label, predicted_amr, EDGE_LABELS_AND_REIFICATIONS[edge_label][0]):
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)
            else:
                # print("third case")
                node_label = label
                if len([inst for inst in predicted_amr.instances() if inst.target == node_label]) >= 1:
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)
        except KeyError as e:
            print("error:", e)
            print(gold_amr.metadata)
            raise e

    def make_results(self):
        if self.category_metadata.subtype == "hand-crafted":
            fun = evaluate_word_disambiguation
        elif self.category_metadata.subtype == "bert":
            fun = evaluate_berts_mouth
        else:
            raise NotImplementedError(f"subtype {self.category_metadata.subtype} not implemented: must be bert or hand-crafted")
        self.gold_amrs, self.predicted_amrs = filter_amrs_for_name(self.category_metadata.subcorpus_filename,
                                                                   self.gold_amrs, self.predicted_amrs)
        successes, sample_size = fun(self.gold_amrs, self.predicted_amrs)
        self.make_and_append_results_row("Recall", EVAL_TYPE_SUCCESS_RATE, [successes, sample_size])
