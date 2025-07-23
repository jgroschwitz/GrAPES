from dataclasses import dataclass, field
from typing import List


@dataclass
class SubcategoryMetadata:
    """
    Stores info about each subcategory
    """
    name: str
    display_name: str
    bunch: int
    tsv: str = None
    subcorpus_filename: str = None
    latex_display_name: str  = None
    other_subcorpus_filename: str = None
    graph_id_column: int = 0
    use_sense: bool = False
    first_row_is_header: bool = False
    # for nodes
    use_attributes: bool = False
    attribute_label: str = None
    metric_label: str = "Recall"
    run_prerequisites: bool = True
    # for edges
    source_column: int = 1
    edge_column: int = 2
    target_column: int = 3
    parent_column: int = None
    parent_edge_column: int = None
    # for named entities, word disambiguation, structural generalisation
    subtype: str = None
    label_column: int = 1
    # for pps and deep recusion with pronouns
    extra_subcorpus_filenames: List[str] = None
    # for gathering results
    additional_fields: List[str] = field(default_factory=list)

    def get_latex_display_name(self):
        if self.latex_display_name is None:
            return self.display_name
        return self.latex_display_name

    def filename_belongs_to_subcategory(self,filename):
        return filename == self.subcorpus_filename or self.extra_subcorpus_filenames is not None and filename in self.extra_subcorpus_filenames


def is_grapes_category_with_testset_data(category_info):
    return  category_info.name == "word_ambiguities_handcrafted"


def is_grapes_category_with_ptb_data(category_info):
    return category_info.name == "unbounded_dependencies"


def is_copyrighted_data(category_info):
    return is_grapes_category_with_testset_data(category_info) or is_grapes_category_with_ptb_data(category_info)


def is_sanity_check(category_info):
    if category_info.subcorpus_filename is None:
        return False
    else:
        return category_info.subcorpus_filename.endswith("sanity_check")
