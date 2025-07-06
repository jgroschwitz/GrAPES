# if everything is in its default location with default names, this script will take you from parser names to:
# CSV files and pickles of results in data/processed/results/from_run_full_evaluation
# LaTeX tables in data/processed/results

echo "##########"
echo "EVALUATING"
echo "##########"


cd ../evaluation/full_evaluation || exit
PYTHONPATH=../.. python run_full_evaluation.py "$@"

printf "\n\n\n##########\n"
echo "   LATEX"
echo "##########"

cd ../../scripts/latex || exit

csvs=()
for parser in "$@"; do csvs+=("../../data/processed/results/from_run_full_evaluation/$parser.csv"); done

PYTHONPATH=../.. python csv2latex.py "${csvs[@]}" -o "../../data/processed/results/latex/$parser/table.tex"

cd ../../data/processed/results/latex || exit

for parser in "$@"; do  cp sandbox.tex $parser/sandbox.tex && cd $parser && pdflatex sandbox.tex $&& cd ..; done

printf "\n\n\n##########\n"
echo "  VULCAN"
echo "##########"

cd ../../../..

for parser in "$@"; do
  printf "\n ** GrAPES corpus ** \n"
python create_vulcan_pickle.py -g corpus/corpus.txt -p "data/processed/parser_outputs/$parser-output/full_corpus.txt" -e -n "$parser" # >> error_analysis/log 2>&1
  printf "\n ** Testset ** "
python create_vulcan_pickle.py -g "data/raw/gold/test.txt" -p "data/processed/parser_outputs/$parser-output/testset.txt" -e -n "$parser"  # >> error_analysis/log 2>&1
echo "Wrote pickles to error_analysis/$parser/vulcan_correct_and_incorrect/"

done