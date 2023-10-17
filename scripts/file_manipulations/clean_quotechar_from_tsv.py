import csv


def main():
    with open("../../corpus/winograd_annotated - Copy.tsv", 'r') as file:  # reentrancies_labeled.tsv
        reader = csv.reader(file, delimiter='\t', quotechar="\"")
        with open("../../corpus/winograd_annotated.tsv", 'w') as output_file:  # reentrancies_labeled_clean.tsv
            writer = csv.writer(output_file, delimiter='\t', quotechar=None)
            for row in reader:
                writer.writerow(row)


if __name__ == "__main__":
    main()
