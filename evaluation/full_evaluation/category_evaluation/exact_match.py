from evaluation.full_evaluation.category_evaluation.category_evaluation import CategoryEvaluation


class ExactMatch(CategoryEvaluation):
    """
    We use this only for Structural Generalisation categories, but could be used for other things.
    Checks exact match and Smatch.
    """

