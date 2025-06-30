from dataclasses import dataclass


@dataclass
class SubcategoryMetadata:
    """
    Stores info about each subcategory
    """
    display_name: str
    tsv: str or None = None
    subcorpus_filename: str or None = None
    latex_display_name: str or None = None
    other_subcorpus_filename: str or None = None
    graph_id_column: int = 0
    use_sense: bool = False
    use_sense_prereq = False
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

    def get_latex_display_name(self):
        if self.latex_display_name is None:
            return self.display_name
        return self.latex_display_name

