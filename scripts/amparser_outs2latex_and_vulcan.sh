# if everything is in its default location with default names, this script will take you amparser output to:
# CSV files and pickles of results in data/processed/results/from_run_full_evaluation
# LaTeX tables in data/processed/results/latex/amparser/ > sandbox.pdf
# Vulcan-readable pickles for error analysis in error_analysis/amparser/



echo "##########"
echo "EVALUATING"
echo "##########"


cd ../evaluation/full_evaluation || exit
PYTHONPATH=../.. python run_full_evaluation.py amparser

ret=$?
if [ $ret -ne 0 ]; then
     exit
fi


printf "\n\n\n##########\n"
echo "   LATEX"
echo "##########"

cd ../../scripts/latex || exit

# with category names and averages in big table:
PYTHONPATH=../.. python csv2latex.py -o "../../data/processed/results/latex/amparser/table.tex" --print_headers "../../data/processed/results/from_run_full_evaluation/amparser.csv" -a "../../data/processed/results/from_run_full_evaluation/amparser_averages.csv"

ret=$?
if [ $ret -ne 0 ]; then
     exit
fi

cd ../../data/processed/results/latex || exit

cp sandbox.tex amparser/sandbox.tex && cd amparser && pdflatex sandbox.tex $&& cd ..

printf "\n\n\n##########\n"
echo "  VULCAN"
echo "##########"

cd ../../../../scripts/visualisation/ || exit

PYTHONPATH=../.. python amparser_all_error_analysis_to_vulcan.py

