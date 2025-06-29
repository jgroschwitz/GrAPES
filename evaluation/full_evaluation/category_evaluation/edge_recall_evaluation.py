from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation, \
    EVAL_TYPE_SUCCESS_RATE
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata
from evaluation.pp_attachment import get_pp_attachment_success_counters



class EdgeRecall(CategoryEvaluation):

    def run_single_evaluation(self, subcategory_info: SubcategoryMetadata):
        self.set_category_metadata(subcategory_info)

        try:
            self.make_results_columns_for_edge_recall()
        except IndexError as e:
            if subcategory_info.subcorpus_filename == "unbounded_dependencies":
                print("Check that corpus/unbounded_dependencies.tsv has 66 rows")
                print("Something may have gone wrong in extending the GrAPES testset with PTB data")
            raise e
        except FileNotFoundError as e:
            if subcategory_info.subcorpus_filename == "unbounded_dependencies":
                print("Check that corpus/unbounded_dependencies.tsv exists")
                print("Something may have gone wrong in extending the GrAPES testset with PTB data")
            raise e
        return self.rows

class NodeRecall(CategoryEvaluation):
    def run_single_evaluation(self, subcategory_info: SubcategoryMetadata):
        self.set_category_metadata(subcategory_info)
        self.make_results_column_for_node_recall()
        if subcategory_info.run_prerequisites:
            self.make_results_column_for_node_recall(prereq=True)
        return self.rows

class PPAttachment(CategoryEvaluation):

    def run_single_evaluation(self, subcategory_info: SubcategoryMetadata):
        self.set_category_metadata(subcategory_info)
        prereqs, unlabeled, recalled, sample_size = get_pp_attachment_success_counters(self.gold_amrs, self.predicted_amrs)
        return [self.make_results_row("Edge recall", EVAL_TYPE_SUCCESS_RATE, [recalled, sample_size]),
                self.make_results_row("Unlabeled edge recall", EVAL_TYPE_SUCCESS_RATE, [unlabeled, sample_size]),
                self.make_results_row("Prerequisites", EVAL_TYPE_SUCCESS_RATE, [prereqs, sample_size])]

class NETypeRecall(CategoryEvaluation):

    def run_single_evaluation(self, subcategory_info: SubcategoryMetadata):
        self.set_category_metadata(subcategory_info)
        self.make_results_for_ne_types()
        return self.rows

class NERecall(CategoryEvaluation):
    def run_single_evaluation(self, subcategory_info: SubcategoryMetadata):
        self.set_category_metadata(subcategory_info)
        self.make_results_for_ne()
        return self.rows