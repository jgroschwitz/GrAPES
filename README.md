# GrAPES

This is the repo for the **Gr**anular **A**MR **P**arsing **E**valuation **S**uite (GrAPES). Our paper "AMR Parsing is Far from Solved: GrAPES, the Granular AMR Parsing Evaluation Suite" has been accepted at EMNLP 2023!

GrAPES provides specialised evaluation metrics and additional data. Throughout the documentation, we distinguish between the AMR 3.0 testset (which you probably already have) and the GrAPES testset, which is our additional data, housed in the `corpus/subcorpora` folder.

# Set up

## Dependencies

GrAPES requires the Python packages `penman`, `prettytable` `statsmodels`, `smatch`, `cryptography>=3.1`.

```commandline
pip install prettytable penman statsmodels smatch cryptography>=3.1
```

GrAPES has been tested with Python 3.8.10.

## Corpus files

GrAPES relies on three sources of data: A) our original data, B) original data based on external licensed corpora (A and B for the GrAPES testset), and C) the AMR testset. GrAPES evaluation can be run on all of them together to obtain scores for all categories, or each separately to obtain scores on only the corresponding categories. (A) requires no additional setup, but (B) and (C) do, see below.

### Obtaining the full GrAPES testset (B).

For licensing reasons, two of the GrAPES categories (Unbounded Dependencies and Word Ambiguities (handcrafted)) are only available if you also have the necessary licenses. You can use GrAPES without that data and skip this setup step, but two categories will be missing. To obtain the full GrAPES corpus, use the following instructions:

The Unbounded Dependencies category is built from Penn Tree Bank sentences. If you have access to the Penn Tree Bank, the following script will add them to the existing GrAPES `corpus.txt` file, where `<ptb_pos_path>` refers to the location of all the POS tagged files in the PTB (in version 2 of the PTB, this is the `tagged` subfolder, in version 3 it is `tagged/pos`).

```commandline
python complete_the_corpus.py -ptb <ptb_pos_path>
```

Twelve of the sentences in the Word Ambiguities (handcrafted) category are AMR 3.0 test set sentences. To add them to the GrAPES `corpus.txt` file, run the following script, where `<amr_test_path>` refers to the AMR 3.0 testset folder (i.e. the folder `data/amr/split/test` in the AMRBank 3.0):

```commandline
python complete_the_corpus.py -amr <amr_test_path>
```

### AMR testset setup (C)

For the evaluation stage (if you want to include the AMR-testset-based categories of GrAPES), GrAPES needs the testset of the AMRBank 3.0 concatenated all into one file (specifically, with the files concatenated in alphabetical order). You can obtain such a concatenation with this script:

```commandline
python concatenate_amr_files.py path/to/original/AMR/testset concatenated/testset/file/name
```

The file `concatenated/testset/file/name` will be created by this script, and in all the documentation below, `concatenated/testset/file/name` refers to that file.


# Usage

## Running your parser

The evaluation scripts use two corpus files, the AMR 3.0 testset and the GrAPES testset provided (and possible extended in step B above). To use GrAPES, you need to generate parser output on both of those datasets. For each dataset, generate one file with AMRs like you would for computing Smatch, i.e. with the AMRs in the same order as the input corpus, and separated by blank lines (that is, the standard AMR corpus format, readable by the `penman` package; we only need the graphs, no metadata like IDs etc. is required).

For the GrAPES testset, simply run your parser on `corpus/corpus.txt` (this file was possibly extended from the version in this repo in setup step B).

For the AMR 3.0 test set, you may already have such an output file. If not, run your parser on the `concatenated/testset/file/name` file created during setup step (C).

If you want to evaluate only on a single category, running your parser on one of the files in `corpus/subcorpora` may be sufficient.

## Evaluation

To run the full evaluation suite, run the following:

```commandline
python evaluate_call_categories.py -gt path/to/AMR/testset -pt path/to/parser/output/AMR/testset -gg corpus/corpus.txt -pg path/to/your/parser/output/GrAPES/corpus.txt 
```

The `-gt` argument is the path to your copy of the AMR testset and the `-pt` argument is the path to your parser output for the AMR testset. The `-gg` argument is the path to the GrAPES file `corpus.txt` and `-pg` is the path to your parser output on that file. This will automatically detect whether you've added the PTB and AMR testset sentences in setup step B.

You can also evaluate on only the AMR testset, or only the GrAPES testset, simply by leaving out the other parameters.

AMR 3.0 testset only:

```commandline
python evaluate_call_categories.py -gt path/to/AMR/testset -pt path/to/parser/output/AMR/testset
```

 GrAPES testset only:

```commandline
python evaluate_call_categories.py -gg corpus/corpus.txt -pg path/to/your/parser/output/GrAPES/corpus.txt 
```

### What do to if you are missing PTB or AMR 3.0

If you don't have AMR 3.0:

* Use only the GrAPES corpus
* The evaluation script will automatically leave out the Word Disambiguation category (which contains some AMR testset sentences)

If you don't have PTB:

* The evaluation script will automatically leave out the Unbounded Dependencies category

### Evaluate on a single category

To evaluate on just one of the 36 categories, use the `evaluate_single_category.py` script and give the name of the category to evaluate, and provide the path to the relevant gold file (`-g`) and the relevant prediction file (`-p`). 

Category names are listed below. The "relevant" gold file is either the path to the AMR testset, the path to the GrAPES gold `corpus.txt` file, or, if you prefer, the GrAPES subcorpus file, such as `adjectives.txt`. Similarly, your parser output can be the full GrAPES `corpus.txt` output, or just the output from running your parser on the one category.

For example, to evaluate on the category Adjectives, which is a GrAPES-only category, either of the following will work:

```commandline
python evaluate_single_category.py -c adjectives -g corpus/corpus.txt -p path/to/parser/full/grapes/output 
```

```commandline
python evaluate_single_category.py -c adjectives -g corpus/adjectives.txt -p path/to/parser/output/adjectives/only 
```

As long as the files have the same number of graphs, the order matches, and they contain the particular category you want, this will work.


To evaluate an AMR testset category, e.g. here the Rare Senses category, run the following.

```commandline
python evaluate_single_category.py -c rare_senses -g path/to/AMR/testset -p path/to/parser/AMR/testset/output
```

#### Category names for the command line

These are also listed if you use the `--help` option.

```
pragmatic_coreference_testset
pragmatic_coreference_winograd
syntactic_gap_reentrancies
unambiguous_coreference
nested_control_and_coordination
nested_control_and_coordination_sanity_check
multiple_adjectives
multiple_adjectives_sanity_check
centre_embedding
centre_embedding_sanity_check
cp_recursion
cp_recursion_sanity_check
cp_recursion_plus_coreference
cp_recursion_plus_coreference_sanity_check
cp_recursion_plus_rc
cp_recursion_plus_rc_sanity_check
cp_recursion_plus_rc_plus_coreference
cp_recursion_plus_rc_plus_coreference_sanity_check
long_lists
long_lists_sanity_check
rare_node_labels
unseen_node_labels
rare_predicate_senses_excl_01
unseen_predicate_senses_excl_01
rare_edge_labels
unseen_edge_labels
seen_names
unseen_names
seen_dates
unseen_dates
other_seen_entities
other_unseen_entities
types_of_seen_named_entities
types_of_unseen_named_entities
seen_andor_easy_wiki_links
hard_unseen_wiki_links
frequent_predicate_senses_incl_01
word_ambiguities_handcrafted
word_ambiguities_karidi_et_al_2021
pp_attachment
unbounded_dependencies
passives
unaccusatives
ellipsis
multinode_word_meanings
imperatives
```


## Looking at example outputs

You may find [Vulcan](https://github.com/jgroschwitz/vulcan) helpful for looking at your parser output and comparing it to the gold graph, when available.

You can Git Clone the repository, and create pickles of the data as follows:

#TODO

You can then view the graphs and sentences side-by-side with Vulcan:

```commandline
python vulcan.py path/to/pickle
```


## Structure of this repository

* All provided corpus files are in `corpus/`, including the main file `corpus.txt`.
* All required python scripts are at the root level
* The evaluation modules are in `evaluation/`
* Code that was used for the paper (but that you don't need to use) is also included. 
  * You may find that running scripts that are not at the root level gives you `PYTHONPATH` trouble. In Mac and Linux, try prepending `PYTHONPATH=./` to the command. In Windows, try to add the parent directory to the Python Path environment variable.


```
GrAPES
├── evaluate_all_categories.py              # main script
├── evaluate_single_category.py             # main script for 1 category
├── corpus                                  # all corpus files, including TSV files used to evaluation
│ └── corpus.txt                            # the full concatenated GrAPES corpus (AMR test set not included)
├── LICENSE
├── README.md
├── docker-compose                          # Docker compose files for AM parser and AMRBART
├── error_analysis                          # TODO
│ └── README.md
├── evaluation                              # all evaluation modules
│ ├── concatenate_amr_files.py
│ ├── corpus_metrics.py
│ ├── create_own_graphs_vulcan_pickle.py
│ ├── full_evaluation                       # full evaluation modules
│ │ ├── category_evaluation                 # evaluation modules by set
│ │ │ ├── category_evaluation.py
│ │ │ ├── i_pragmatic_reentrancies.py
│ │ │ ├── ii_unambiguous_reentrancies.py
│ │ │ ├── iii_structural_generalization.py
│ │ │ ├── iv_rare_unseen_nodes_edges.py
│ │ │ └── v_names_dates_etc.py
│ │ │ ├── vi_entity_classification_and_linking.py
│ │ │ ├── vii_lexical_disambiguation.py
│ │ │ ├── viii_attachments.py
│ │ │ ├── ix_nontrivial_word2node_relations.py
│ │ ├── corpus_statistics.py
│ │ ├── run_full_evaluation.py
│ │ └── wilson_score_interval.py
│ └── testset                               # evaluation modules for the AMR test set categories
├── grammars                                # Alto grammars for structural generalisation
├── scripts
│ ├── full_evaluation.sh                    # script we used for the paper
│ ├── file_manipulations                    # various scripts for changing files
│ ├── preprocessing                         # preprocessing scripts for AM parser and AMRBART
│ └── single_evaluation.sh                  
└── amrbank_analysis                        # various scripts and modules used in the creation of GrAPES
```

## Credits

This work builds on (and contains parts of) the [Winograd Schema Challenge](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html), which is published under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

This work also builds on the [Putting Words into BERT's Mouth](https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth) corpus.
