from dataclasses import dataclass
from typing import List


@dataclass
class SubcategoryMetadata:
    """
    Stores info about each subcategory
    """
    name: str
    display_name: str
    tsv: str or None = None
    subcorpus_filename: str or None = None
    latex_display_name: str or None = None
    other_subcorpus_filename: str or None = None
    graph_id_column: int = 0
    use_sense: bool = False
    first_row_is_header: bool = False
    # for nodes
    use_attributes: bool = False
    attribute_label: str or None = None
    metric_label: str = "Label recall"
    run_prerequisites: bool = True
    # for edges
    source_column: int or None = 1
    edge_column: int or None = 2
    target_column: int or None = 3
    parent_column: int or None = None
    parent_edge_column: int or None = None
    # for named entities, word disambiguation, structural generalisation
    subtype: str or None = None
    label_column: int = 1
    # for pps and deep recusion with pronouns
    extra_subcorpus_filenames: List[str] or None = None

    def get_latex_display_name(self):
        if self.latex_display_name is None:
            return self.display_name
        return self.latex_display_name

    def filename_belongs_to_subcategory(self,filename):
        return filename == self.subcorpus_filename or self.extra_subcorpus_filenames is not None and filename in self.extra_subcorpus_filenames


def is_grapes_category_with_testset_data(category_info):
    return  category_info.subcorpus_filename == "word_disambiguation"


def is_grapes_category_with_ptb_data(category_info):
    return category_info.subcorpus_filename == "unbounded_dependencies"


def is_copyrighted_data(category_info):
    return is_grapes_category_with_testset_data(category_info) or is_grapes_category_with_ptb_data(category_info)


def is_sanity_check(category_info):
    return category_info.subcorpus_filename.endswith("sanity_check")
