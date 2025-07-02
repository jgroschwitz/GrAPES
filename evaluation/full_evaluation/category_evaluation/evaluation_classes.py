from debugpy.server.cli import TARGET

from evaluation.graph_matcher import check_fragment_existence
from evaluation.file_utils import read_label_tsv
from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE, PREREQS
from evaluation.novel_corpus.pp_attachment import get_pp_attachment_success_counters
from evaluation.util import get_node_by_name


class PPAttachment(CategoryEvaluation):
    def __init__(self, gold_amrs, predicted_amrs, root_dir, info, predictions_directory=None):
        """
        Pragmatic attachments of ambiguous PPs
        PP Attachments come from multiple files, so if they're not already in the given graphs, we try to get them.
        """
        super().__init__(gold_amrs, predicted_amrs, root_dir, info, predictions_directory)
        # if we read in the unused PP directory instead of the whole full_cprpus, replace it with the real ones
        # These have ids pp_attachment_n
        if self.gold_amrs[0].metadata['id'].startswith(self.category_metadata.subcorpus_filename):
            print("Reading in additional files")
            self.gold_amrs, self.predicted_amrs = self.get_additional_graphs(read_in=True)
        if len(self.gold_amrs) != len(self.predicted_amrs) or len(self.gold_amrs) ==0:
            raise Exception("Different number of AMRs or 0")

    def make_results(self):
        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.gold_amrs, self.predicted_amrs)
        assert sample_size>0, "No results found!"
        rows = [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]
        self.rows.extend(rows)


class SubgraphRecall(CategoryEvaluation):
    """For multinode word meanings like "teacher" = person <- teach-01"""

    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):
        if check_fragment_existence(target, predicted_amr):
            self.add_success(gold_amr, predicted_amr)
        else:
            self.add_fail(gold_amr, predicted_amr)


class EllipsisRecall(CategoryEvaluation):
    """Find two instances of a subgraph where one is elided in the sentence"""

    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):

        count = len([instance for instance in predicted_amr.instances() if instance.target == target])
        if count >= 1:
            self.add_success(gold_amr, predicted_amr, PREREQS)
        else:
            self.add_fail(gold_amr, predicted_amr, PREREQS)
        if count >= 2:
            self.add_success(gold_amr, predicted_amr)
        else:
            self.add_fail(gold_amr, predicted_amr)


TARGET = "modulo_imperative_target"

class ImperativeRecall(CategoryEvaluation):
    def __init__(self, gold_amrs, predicted_amrs, root_dir, info, predictions_directory=None, do_error_analysis=False):
        super().__init__(gold_amrs, predicted_amrs, root_dir, info, predictions_directory, do_error_analysis)

    def read_tsv(self):
        return read_label_tsv(root_dir=self.root_dir, tsv_file_name=self.category_metadata.tsv, columns=[1, 2, 3])

    def update_results(self, gold_amr, predicted_amr, target, predictions_for_comparison=None):
        verb_label, imperative_target_edge_label, imperative_target = target
        matching_instances = [instance for instance in predicted_amr.instances() if instance.target == verb_label]
        if len(matching_instances) >= 1:
            self.add_success(gold_amr, predicted_amr, PREREQS)
            with_imperative = [instance for instance in matching_instances if
                               len(predicted_amr.attributes(source=instance.source, role=":mode",
                                                            target="imperative")) > 0]
            if len(with_imperative) >= 1:
                self.add_success(gold_amr, predicted_amr, TARGET)
                found_imperative_target = False
                for instance in with_imperative:
                    for edge in predicted_amr.edges(source=instance.source,
                                                    role=":" + imperative_target_edge_label):
                        imperative_target_node = get_node_by_name(edge.target, predicted_amr)
                        if imperative_target_node is not None and imperative_target_node.target == imperative_target:
                            found_imperative_target = True
                if found_imperative_target:
                    self.add_success(gold_amr, predicted_amr)
                else:
                    self.add_fail(gold_amr, predicted_amr)
            else:
                self.add_fail(gold_amr, predicted_amr)
                self.add_fail(gold_amr, predicted_amr, TARGET)
        else:
            self.add_fail(gold_amr, predicted_amr, PREREQS)
            self.add_fail(gold_amr, predicted_amr, TARGET)
            self.add_fail(gold_amr, predicted_amr)



