import sys


def main(args):
    """
    Run from main directory
    :param args:
    :return:
    """
    with open(f"corpus/{args[1]}.txt", "r") as f:
        with open(f"corpus/{args[1]}_cailam.txt", "w") as w:
            only_empties = True
            for line in f.readlines():
                skip_line = line.startswith("#") and not (line.startswith("# ::") or line.startswith("#::"))
                early_empty_line = line.strip() == "" and only_empties
                skip_line = skip_line or early_empty_line
                if not skip_line:
                    only_empties = False
                    w.write(line)


if __name__ == "__main__":
    main(sys.argv)