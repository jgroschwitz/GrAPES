from penman import load


def main():
    amrs = load("../../corpus/word_disambiguation1.txt")
    for amr in amrs:
        print(amr.metadata["id"])
    print(len(amrs))


if __name__ == "__main__":
    main()
