import os

import penman


def main():

    # list all files in ../../corpus
    for filename in os.listdir("../../corpus"):
        if file_is_amr_corpus("../../corpus/"+filename):
            add_ids_to_file("../../corpus/"+filename)


def file_is_amr_corpus(filename):
    if not filename.endswith(".txt"):
        return False
    # check if the file contains the string "::snt"
    with open(filename, "r") as f:
        for line in f:
            if "::snt" in line:
                return True


def add_ids_to_file(filename):
    filename_infix = filename.split("/")[-1].split(".")[0]
    # filename_infix.replace(" ", "_") + "_" + str(i)
    with open(filename, "r") as f:
        lines = f.readlines()

    with open(filename, "w") as f:
        i = 0
        id_seen = False
        for line in lines:
            if "::id" in line:
                id_seen = True
            if line.strip() == "":
                id_seen = False
            if line.startswith("("):
                if not id_seen:
                    f.write(f"# ::id {filename_infix}_{i}\n")
                i += 1
            f.write(line)


if __name__ == '__main__':
    main()
