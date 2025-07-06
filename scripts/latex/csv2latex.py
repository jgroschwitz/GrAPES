import argparse
import os
import sys
import csv

from evaluation.full_evaluation.category_evaluation.category_metadata import bunch_number2name


def main():
    """
    This returns a longtable environment to be used in a LaTeX document. You need to include the following in your
    document preamble (use a single backslash for \\usepockage; just need to escape here):
    \\usepackage{longtable}
    \\usepackage{xcolor, colortbl}
    \definecolor{lightlightlightgray}{gray}{0.95}
    \newcommand{\successScore}[4]{#1 \scriptsize\textcolor{gray}{#4[#2,#3]}}

    also, you probably want to print the whole table with \small in front.

    TODO: on Jonas' Laptop, the current setup adds a weird "2" to the Unaccusatives dataset name.

    """
    default_tex_file = "../../data/processed/results/latex/table.tex"

    parser = argparse.ArgumentParser(description="Turn CSV into LaTeX table")
    parser.add_argument("-o", "--output_file", type=str, help=f"Path to the output file (default {default_tex_file})", default=default_tex_file)
    parser.add_argument("csv_files", type=str, nargs="+", help="Paths to the CSV files")
    parser.add_argument("--print_headers", action="store_true", help="Print a header for each set of categories")
    parser.add_argument("-a", "--csv_averages_files", type=str, nargs="+", help="Paths to the CSV average files")
    args = parser.parse_args()

    dir_path = args.output_file.split("/")[:-1]

    os.makedirs("/".join(dir_path), exist_ok=True)

    # read csv files
    csv_contents = []
    for csv_file in args.csv_files:
        with open(csv_file, "r") as f:
            csv_reader = csv.reader(f)
            csv_contents.append(list(csv_reader)[1:]) # skip header

    if args.csv_averages_files is not None:
        averages_contents = []
        for averages_file in args.csv_averages_files:
            with open(averages_file, "r") as f:
                csv_reader = csv.reader(f)
                averages_contents.append(list(csv_reader))
        transposed_averages_contents = [list(i) for i in zip(*averages_contents)]

    else:
        transposed_averages_contents = None

    names = [os.path.basename(csv_file)[:-4] for csv_file in args.csv_files]

    transposed_csv_contents = [list(i) for i in zip(*csv_contents)]
    # print(csv_contents)

    write_full_table(args, names, transposed_averages_contents, transposed_csv_contents)

    if args.csv_averages_files is not None:
        write_compact_table(args, names, transposed_averages_contents)


def write_full_table(args, names, transposed_averages_contents, transposed_csv_contents):
    head_column = "Set & Category & Metric & " + " & ".join([name.replace("_", "") for name in names]) + " & \\#"
    table_columns = "{l | l | l  | " + " | ".join(["c"] * len(args.csv_files)) + " | r }"
    header = "\\begin{longtable}" + table_columns + "\n\t" + head_column + "\\\\\\hline\n"
    closer = "\\end{longtable}"
    print_header = args.print_headers
    set_id = 0
    old_dataset_name = ""
    with open(args.output_file, "w") as f:
        f.write(header)
        shade_row = True
        for zipped_csv_row in transposed_csv_contents:
            old_set_id = set_id
            set_id = int(zipped_csv_row[0][0])
            # is_header = zipped_csv_row[0][2] == ""
            # if is_header and not print_header:
            #     set_id = int(set_id) - 1
            #     continue

            first_row_of_the_bunch = set_id != old_set_id
            set_id_to_print = set_id if first_row_of_the_bunch else ""
            dataset_name = zipped_csv_row[0][2]
            dataset_name_to_print = dataset_name if dataset_name != old_dataset_name else ""
            # if is_header:
            #     set_id_to_print = f"\\textbf{{{set_id_to_print}}}"
            #     dataset_name_to_print = f"\\textbf{{{dataset_name_to_print}}}"
            if first_row_of_the_bunch and transposed_averages_contents is not None and print_header:
                # insert a header for the bunch with its name and the average score
                scores = [entry[1] for entry in transposed_averages_contents[set_id - 1]]
                set_name = zipped_csv_row[0][1]
                bunch_id_to_print = f"\\textbf{{{set_id_to_print}}}"
                bunch_name_to_print = f"\\textbf{{{set_name}}}"
                metric_name = "Average"
                shade_row = not shade_row
                shading_prefix = "\\rowcolor{lightlightlightgray}" if shade_row else ""
                latex_line = f"\t{shading_prefix}{bunch_id_to_print} & {bunch_name_to_print} & {metric_name}"
                for s in scores:
                    latex_line += f" & {f3(str(s))}"
                f.write(latex_line + "&\\\\\n")
                set_id_to_print = ""  # printing set id in header, so no need in categories

            old_dataset_name = dataset_name
            metric_name = zipped_csv_row[0][3]
            scores = [entry[4] for entry in zipped_csv_row]
            lower_bounds = [entry[5] for entry in zipped_csv_row]
            upper_bounds = [entry[6] for entry in zipped_csv_row]
            count = zipped_csv_row[0][7]

            is_sanity_check_row = "sanity" in dataset_name.lower()
            is_prereq_row = "prereq" in metric_name.lower()
            is_smatch_row = "smatch" in metric_name.lower()
            is_still_long_lists = "precision" in metric_name.lower() or "unseen" in metric_name.lower()
            is_same_category = is_smatch_row or is_prereq_row or is_sanity_check_row or is_still_long_lists
            if is_same_category:
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


def write_compact_table(args, names, transposed_averages_contents):
    head_column = " Set & " + " & ".join([name.replace("_", "") for name in names])
    table_columns = "{l  | " + " | ".join(["c"] * len(args.csv_files)) + " }"
    header = "\\begin{tabular}" + table_columns + "\n\t" + head_column + "\\\\\\hline\n"
    closer = "\\end{tabular}"
    with open(args.output_file, "a") as f:
        f.write(header)
        for i, zipped_csv_row in enumerate(transposed_averages_contents):
            set_id = i + 1
            name = bunch_number2name[set_id]
            scores = [entry[1] for entry in zipped_csv_row]

            latex_line = f"\t{set_id}.\\ {name} "
            for s in scores:
                latex_line += f" & {f3(s)}"
            latex_line += f"\\\\\n"
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
