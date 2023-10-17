import sys
from nltk.tokenize import word_tokenize

if __name__ == "__main__":
    # get first argument: dataset name
    dataset_name = sys.argv[1]
    with open(f"corpus/{dataset_name}.txt") as f:
        with open(f"corpus/{dataset_name}_tokenized_input.txt", "w") as f2:
            for line in f:
                if line.startswith("# ::snt"):
                    sentence = line.split("::snt ")[1].strip()
                    tokenized_sentence = " ".join(word_tokenize(sentence))
                    f2.write(tokenized_sentence + "\n")
