import sys
import csv


def main():
    """
    Usage: python csv2latex.py output_file csv_files
    Example: python csv2latex.py output.tex parser1.csv parser2.csv parser3.csv

    This returns a tabular environment to be used in a LaTeX document. You need to include the following in your
    document preamble (use a single backslash for \\usepockage; just need to escape here):
    \\usepackage{xcolor, colortbl}
    \definecolor{lightlightlightgray}{gray}{0.95}
    \newcommand{\successScore}[4]{#1 \scriptsize\textcolor{gray}{#4[#2,#3]}}


    also, you probably want to print the whole table with \small in front.


    TODO: on Jonas' Laptop, the current setup adds a weird "2" to the Unaccusatives dataset name.

    """
    # arguments: output_file, csv_files
    output_file = sys.argv[1]
    csv_files = sys.argv[2:]

    # read csv files
    csv_contents = []
    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            csv_reader = csv.reader(f)
            csv_contents.append(list(csv_reader))

    transposed_csv_contents = [list(i) for i in zip(*csv_contents)]
    # print(csv_contents)

    head_column = "Set ID & Dataset & Metric & " + " & ".join([name.replace("_", "") for name in csv_files]) + " & \\#"
    table_columns = "{l | l | l  | " + " | ".join(["c"] * len(csv_files)) + " | r }"
    header = "\\begin{tabular}" + table_columns + "\n\t" + head_column + "\\\\\\hline\n"
    closer = "\\end{tabular}"

    set_id = 0
    old_dataset_name = ""
    with open(output_file, "w") as f:
        f.write(header)
        shade_row = True
        for zipped_csv_row in transposed_csv_contents:
            old_set_id = set_id
            set_id = zipped_csv_row[0][0]
            set_id_to_print = set_id if set_id != old_set_id else ""
            dataset_name = zipped_csv_row[0][1]
            dataset_name_to_print = dataset_name if dataset_name != old_dataset_name else ""
            old_dataset_name = dataset_name
            metric_name = zipped_csv_row[0][2]
            scores = [entry[3] for entry in zipped_csv_row]
            lower_bounds = [entry[4] for entry in zipped_csv_row]
            upper_bounds = [entry[5] for entry in zipped_csv_row]
            count = zipped_csv_row[0][6]

            is_sanity_check_row = "sanity" in dataset_name.lower()
            is_prereq_row = "prereq" in metric_name.lower()
            if is_sanity_check_row or is_prereq_row:
                pass
            else:
                shade_row = not shade_row

            shading_prefix = "\\rowcolor{lightlightlightgray}" if shade_row else ""
            latex_line = f"\t{shading_prefix}{set_id_to_print} & {dataset_name_to_print} & {metric_name}"
            for s, l, u in zip(scores, lower_bounds, upper_bounds):
                # check if bounds l and u are actually numbers
                if l.replace(".", "", 1).isdigit() and u.replace(".", "", 1).isdigit():
                    latex_line += f" & \\successScore{{{f3(s)}}}{{{f2(l)}}}{{{f3(u)}}}{{\\phantom{{1}}}}"
                else:
                    latex_line += f" & {f3(s)}"
            latex_line += f" & {count}\\\\\n"

            f.write(latex_line)
        f.write(closer)


def f2(number_string):
    if len(number_string) == 1:
        number_string = "0" + number_string
    return number_string

def f3(number_string):
    number_string = f2(number_string)
    if len(number_string) == 2:
        number_string = "\\phantom{1}" + number_string
    return number_string

if __name__ == "__main__":
    main()
