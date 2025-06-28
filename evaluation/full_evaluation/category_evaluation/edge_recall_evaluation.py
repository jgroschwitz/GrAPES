from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation
from evaluation.full_evaluation.category_evaluation.subcategory_info import SubcategoryMetadata


class EdgeRecall(CategoryEvaluation):

    def run_single_evaluation(self, subcategory_info: SubcategoryMetadata):
        self.set_category_metadata(subcategory_info)
        self.make_results_columns_for_edge_recall()
        return self.rows

class NodeRecall(CategoryEvaluation):
    def run_single_evaluation(self, subcategory_info: SubcategoryMetadata):
        self.set_category_metadata(subcategory_info)
        self.make_results_column_for_node_recall()
        if subcategory_info.run_prerequisites:
            self.make_results_column_for_node_recall(prereq=True)
        return self.rows