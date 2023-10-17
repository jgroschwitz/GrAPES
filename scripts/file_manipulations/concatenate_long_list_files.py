

def main():
    dataset_names = ["i_counted", "please_buy", "she_visited_countries"]
    with open("../../corpus/long_lists.txt", "w") as outfile:
        for dataset_name in dataset_names:
            with open(f"../../corpus/{dataset_name}.txt") as infile:
                outfile.write(infile.read())

    with open("../../corpus/long_lists_sanity_check.txt", "w") as outfile:
        for dataset_name in dataset_names:
            with open(f"../../corpus/{dataset_name}_sanity_check.txt") as infile:
                outfile.write(infile.read())


if __name__ == "__main__":
    main()