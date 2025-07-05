# GrAPES

This is the repo for the **Gr**anular **A**MR **P**arsing **E**valuation **S**uite (GrAPES). [Our paper "AMR Parsing is Far from Solved: GrAPES, the Granular AMR Parsing Evaluation Suite" was published in the EMNLP 2023 proceedings](https://aclanthology.org/2023.emnlp-main.662/).

GrAPES provides specialised evaluation metrics and additional data. Throughout the documentation, we distinguish between the AMR 3.0 testset (which you probably already have) and the GrAPES testset, which is our additional data, housed in the `corpus/subcorpora` folder.

# Set up

## Dependencies

GrAPES requires the Python packages `penman`, `prettytable` `statsmodels`, `smatch`, `cryptography>=3.1`.

```commandline
pip install prettytable penman statsmodels smatch "cryptography>=3.1"
```

GrAPES has been tested with Python 3.8.10 and 3.10.13.

## Corpus files

GrAPES relies on three sources of data: A) our original data, B) the AMR testset, and C) original data based on external licensed corpora (A and C form the GrAPES testset). GrAPES evaluation can be run on all of them together to obtain scores for all categories, or each separately to obtain scores on only the corresponding categories. (A) requires no additional setup, but (B) and (C) do, see below.

### AMR testset setup (B)

For the evaluation stage (if you want to include the AMR-testset-based categories of GrAPES), GrAPES needs the testset of the AMRBank 3.0 concatenated all into one file (specifically, with the files concatenated in alphabetical order). You can obtain such a concatenation with this script:

```commandline
python concatenate_amr_files.py path/to/original/AMR/testset concatenated/testset/file/name
```

The file `concatenated/testset/file/name` will be created by this script, and in all the documentation below, `concatenated/testset/file/name` refers to that file.



### Obtaining the full GrAPES testset (C).

For licensing reasons, two of the GrAPES categories (Unbounded Dependencies and Word Ambiguities (handcrafted)) are only available if you also have the necessary licenses. You can use GrAPES without that data and skip this setup step, but two categories will be missing. To obtain the full GrAPES corpus, use the following instructions:

#### Unbounded Dependencies

The Unbounded Dependencies category is built from Penn Tree Bank sentences. If you have access to the Penn Tree Bank, the following script will add them to the existing GrAPES `corpus.txt` file, where `<ptb_pos_path>` refers to the location of all the POS tagged files in the PTB (in version 2 of the PTB, this is the `tagged` subfolder, in version 3 it is `tagged/pos`).

```commandline
python complete_the_corpus.py -ptb <ptb_pos_path>
```
**Troubleshooting:** These are encrypted in `corpus/copyrighted_data`. A version of the TSV file without the sentences is also provided in `corpus/unbounded_dependencies_stripped.tsv` from which the annotations can probably be reconstructed if you have the PTB but the decryption fails. IDs include PTB filenames.

#### Word Ambiguities

Twelve of the sentences in the Word Ambiguities (handcrafted) category are AMR 3.0 test set sentences. To add them to the GrAPES `corpus.txt` file, run the following script, where `<amr_test_path>` refers to the AMR 3.0 concatenated test set file folder (see step B)):

```commandline
python complete_the_corpus.py -amr <amr_test_path>
```
**Troubleshooting:** The file without the copyrighted sentences is `corpus/word_ambiguities_clean.txt`. To create the right files, replace `(removed -- see documentation)` with the sentences (IDs are in the file), save it as `corpus/subcorpora/word_disambiguation.txt`, and add the entries to `corpus/corpus.txt`.

# Usage

## Running your parser

The evaluation scripts use two corpus files, the AMR 3.0 testset and the GrAPES testset provided (and possible extended in step C above). To use GrAPES, you need to generate parser output on both of those datasets. For each dataset, generate one file with AMRs like you would for computing Smatch, i.e. with the AMRs in the same order as the input corpus, and separated by blank lines (that is, the standard AMR corpus format, readable by the `penman` package; we only need the graphs, no metadata like IDs etc. is required).

For the GrAPES testset, simply run your parser on `corpus/corpus.txt` (this file was possibly extended from the version in this repo in setup step C).

For the AMR 3.0 test set, you may already have such an output file. If not, run your parser on the `concatenated/testset/file/name` file created during setup step (B).

If you want to evaluate only on a single category, running your parser on one of the files in `corpus/subcorpora` may be sufficient.

## Evaluation

To run the full evaluation suite, run the following:

```commandline
python evaluate_all_categories.py -gt path/to/AMR/testset -pt path/to/parser/output/AMR/testset -pg path/to/your/parser/output/GrAPES/corpus.txt 
```
Where the arguments are:

* `-gt`: path to your copy of the AMR testset 
* `-pt`: path to your parser output for the AMR testset 
* `-pg`: path to your parser output on the GrAPES corpus in `corpus/corpus.tex`. This will automatically detect whether you've added the PTB and AMR testset sentences in setup step B.
* If your GrAPES gold file `corpus.txt` is not in `corpus/corpus.tex`, add the argument `-gg path/to/your/gold/corpus.txt` 

You can also evaluate on only the AMR testset, or only the GrAPES testset, simply by leaving out the other parameters.

AMR 3.0 testset only:

```commandline
python evaluate_all_categories.py -gt path/to/AMR/testset -pt path/to/parser/output/AMR/testset
```

 GrAPES testset only:

```commandline
python evaluate_all_categories.py -pg path/to/your/parser/output/GrAPES/corpus.txt 
```

Additional options include:
* `--parser_name`: name your parser for more specific output file naming
* `--smatch`: running Smatch on all subcategories (slow),
* `--error_analysis` writing the graph IDs of successes and failures to pickled dictionaries,
* `--all_metrics`: printing Smatch results on Structural Generalisation categories and unlabelled edge recall on appropriate categories (not included in the GrAPES paper)

### What do to if you are missing PTB or AMR 3.0

If you don't have AMR 3.0:

* Use only the GrAPES corpus
* The evaluation script will automatically leave out the Word Disambiguation category (which contains some AMR testset sentences)

If you don't have PTB:

* The evaluation script will automatically leave out the Unbounded Dependencies category

### Evaluate on a single category

To evaluate on just one of the 36 categories, use the `evaluate_single_category.py` script and give the name of the category to evaluate, and provide the path to the  relevant prediction file (`-p`).

Category names are listed below. The "relevant" predictions file is the path to your parser's output on corresponding corpus: the AMR testset, GrAPES `corpus.txt` file, or, if you prefer, the GrAPES subcorpus file, such as `adjectives.txt`. 

If your gold corpus files are not in `corpus/corpus.tex` and `corpus/subcorpora`, include the path to the gold file with option `-g`.

For example, to evaluate on the category Adjectives, which is a GrAPES-only category, either of the following will work:

```commandline
python evaluate_single_category.py -c adjectives  -p path/to/parser/full/grapes/output 
```

```commandline
python evaluate_single_category.py -c adjectives -p path/to/parser/output/adjectives/only 
```

As long as the files have the same number of graphs, the order matches, and they contain the particular category you want, this will work.

To evaluate an AMR testset category, e.g. here the Rare Senses category, run the following.

```commandline
python evaluate_single_category.py -c rare_senses -p path/to/parser/AMR/testset/output
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

The files each category uses are in `evaluation`

## Running evaluations on multiple parsers at once

You can use `evaluation/full_evaluation/run_all_evaluations.py` if you set yourself up as follows:

* In `data/raw/gold`, place a copy of your concatenated AMR 3.0 testset and call it `test.txt`
* For each parser:
  * Choose a name e.g. `"my_parser"` 
  * create a directory in `data/processed/parser_outputs` called `my_parser-outputs`
  * place all output files here:
    * the output of the full grapes corpus as `full_corpus.txt`
    * any single-category output files
    * the output on the AMR 3.0 testset as `testset.txt`
* For Python Path reasons, running this as a script can be hard. You have (at least) two choices:
  1. Edit the `parser_names` variable at the top of the file to be your parser names, and just run the file from within your IDE
  2. Run it as a script from its folder, with python path set to two directories up (`../..`). For each parser you want to include, include a command line argument For example:

```commandline
PYTHONPATH=../../ python run_full_evaluation.py amparser amrbart
```

## Details about the construction of each category

The appendix of the [paper](https://aclanthology.org/2023.emnlp-main.662/) (also in documents/grapes.pdf) provides extensive details for each of the 36 categories.

## Looking at example outputs

You may find [Vulcan](https://github.com/jgroschwitz/vulcan) helpful for looking at your parser output and comparing it to the gold graph, when available. Git Clone the repository, and install the dependencies. 

From your GrAPES main folder, create pickles of the data. This works for any pair of files with predicted and gold graphs in the same order.

```commandline
python create_vulcan_pickle.py path/to/prediction/file path/to/gold/file path/to/output.pickle
```
For example, for the adjectives subcorpus, you could have something like:

```commandline
python create_vulcan_pickle.py ../parser_outputs/subcorpora/adjectives_predictions.txt corpus/subcorpora/adjectives.txt error_analysis/adjectives.pickle
```

You can then view the graphs and sentences side-by-side with Vulcan from your Vulcan folder (not from your GrAPES folder!):

```commandline
python launch_vulcan.py path/to/pickle
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
├── concatenate_amr_files.py                # for setup
├── complete_the_corpus.py                  # for setup
├── create_vulcan_pickle.py                 # for visualising predicted/gold pairs
├── corpus                                  # all GrAPES corpus files, including TSV files used for evaluation
│ ├── subcorpora                            # all GrAPES AMR files (AMR test set not included)
│ └── corpus.txt                            # the full concatenated GrAPES corpus (AMR test set not included)
├── LICENSE
├── README.md
├── docker-compose                          # Docker compose files for AM parser and AMRBART
├── error_analysis                          # a good place for Vulcan pickles
│ └── README.md
── documents
│   └── grapes.pdf                          # the paper, including detailed appendix re categories
├── evaluation                              # all evaluation modules
│ ├── corpus_metrics.py
│ ├── full_evaluation                       # full evaluation modules
│ │ ├── category_evaluation                 # category evaluation modules
│ │ │ ├── category_evaluation.py            # abstract class
│ │ │ ├── evaluation_classes.py             # classes for specific evaluations
│ │ │ ├── subcategory_info.py               # defines dataclass to store info about each subcategory for evaluation
│ │ │ └── category_metadata.py              # subcategory info by category
│ │ ├── corpus_statistics.py
│ │ ├── run_full_evaluation.py              # runs evaluations on multiple parsers
│ │ └── wilson_score_interval.py
│ └── testset                               # evaluation modules for the AMR test set categories
├── grammars                                # Alto grammars for structural generalisation
├── scripts
│ ├── full_evaluation.sh                    # script we used for the paper (may be obsolete)
│ ├── file_manipulations                    # various scripts for changing files
│ ├── latex                                 # converts csv outs from run_all_evaluations to LaTeX table
│ ├── preprocessing                         # preprocessing scripts for AM parser and AMRBART
│ └── single_evaluation.sh                  
└── amrbank_analysis                        # various scripts and modules used in the creation of GrAPES
```

## Credits

Authors: Jonas Groschwitz, Shay B. Cohen, Lucia Donatelli, & Meaghan Fowlie

This work builds on (and contains parts of) the [Winograd Schema Challenge](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html), which is published under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

This work also builds on the [Putting Words into BERT's Mouth](https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth) corpus.


```

```