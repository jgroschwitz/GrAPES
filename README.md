# GrAPES

This is the repo for the Granular AMR Parsing Evaluation Suite (GrAPES). Our paper "AMR Parsing is Far from Solved: GrAPES, the Granular AMR Parsing Evaluation Suite" has been accepted at EMNLP 2023! Documentation of this repository will come soon.

#TODO make grapes vs grapes set clearer

## Set up

The evaluation scripts use two corpus files, the AMR 3.0 testset and the GrAPES test set provided. The AMR 3.0 test set is the files concatenated in alphabetical order; to get this file, run the following script. Provide the path to the testset folder and the path, including the filename, you want to write the concatenated testset file to.

Throughout the rest of this documentation, `path/to/AMR/testset` means the path to the output file.

```commandline
python #TODO
```

If your parser outputs multiple files, as long as their alphanumeric order is the same, this script should work on your files as well.

If you do not have access to the AMR 3.0 test set, you can still use parts of GrAPES. Instructions are provided below.

For licensing reasons, two of the GrAPES categories are only available if you also have the necessary licenses. The Unbounded Dependencies category is built from Penn Tree Bank sentences. If you have access to the Penn Tree Bank, the following script will add them to the existing GrAPES `corpus.txt` file.

```commandline
python #TODO path/to/your/PTB
```

Twelve of the sentences in the Word Disambiguation category are AMR 3.0 test set sentences. To add them to the GrAPES `corpus.txt` file, run the following script:

```commandline
python #TODO path/to/AMR/testset
```

### Dependencies

GrAPES requires the Python packages `penman`, `prettytable` `statsmodels`, `smatch`.

```commandline
pip install prettytable penman statsmodels smatch
```

## Usage

### Evaluate on all categories

To run the full evaluation suite, run the following:

```commandline
python evaluate_call_categories.py -gt path/to/AMR/testset -pt path/to/parser/output/AMR/testset -gg path/to/gold/gRAPES/corpus.txt -pg path/to/your/parser/output/gRAPES/corpus.txt 
```

You can run it on only the AMR testset, on the GrAPES-provided corpus, or both.

To evaluate with GrAPES on categories drawn from the AMR testset, provide the path to your copy of the AMR testset (`-gt`) and the path to your parser output for the AMR testset (`-pt`).

To evaluate on the additional test items provided by GrAPES, provide the path to the GrAPES file `corpus.txt` (`-gg`) and your parser output on that file (`-pg`).

### What do to if you are missing PTB or AMR 3.0

If you don't have AMR 3.0:

* Use only the GrAPES corpus
* The evaluation script will automatically leave out the Word Disambiguation category (which contains some AMR testset sentences)

If you don't have PTB:

* The evaluation script will automatically leave out the Unbounded Dependencies category

### Evaluate on a single category

To evaluate on just one of the 36 categories, give the name of the category to evaluate, and provide the path to the relevant gold file (`-g`) and the relevant prediction file (`-p`). 

Category names are listed below. The "relevant" gold file is either the path to the AMR testset, the path to the GrAPES gold `corpus.txt` file, or, if you prefer, the GrAPES subcorpus file, such as `adjectives.txt`. Similarly, your parser output can be the full GrAPES `corpus.txt` output, or just the output from running your parser on the one category.

For example, to evaluate on the category Adjectives, which is a GrAPES-only category, either of the following will work:

```commandline
python evaluate_single_category.py -c adjectives -g corpus/corpus.txt -p path/to/parser/full/grapes/output 
```

```commandline
python evaluate_single_category.py -c adjectives -g corpus/adjectives.txt -p path/to/parser/output/adjectives/only 
```

As long as the files have the same number of graphs, the order matches, and they contain the particular category you want, this will work.


To evaluate on the category Rare Senses, an AMR testset category, run the following.

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

You may find Vulcan #TODO helpful for looking at your parser output and comparing it to the gold graph, when available.

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
