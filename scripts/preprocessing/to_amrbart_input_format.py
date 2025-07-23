import os
import sys

def main(args):
    dataset_name = args[1]
    with open(f"corpus/{dataset_name}.txt") as f:
        out_dir = f"data/raw/parser_output/amrbart"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{dataset_name}_amrbart_input.jsonl", "w") as f2:
            for line in f:
                if line.startswith("# ::snt "):
                    try:
                        sent = line.split("::snt ")[1].strip()
                        sent = sent.replace("\"", "\\\"")
                        f2.write(f"{{\"sent\": \"{sent}\", \"amr\": \"\"}}\n")
                    except:
                        print(line)


if __name__ == "__main__":
    main(sys.argv)
