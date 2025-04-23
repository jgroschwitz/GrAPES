import argparse
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.concatkdf import ConcatKDFHash

from scripts.file_manipulations.write_copyrighted_data import write_ptb_data, update_from_amr_testset

with open("corpus/copyrighted_data/encoded_unbounded_dep_txt.txt", "r") as f:
    ptb_txt_content = f.read()

with open("corpus/copyrighted_data/encoded_unbounded_dep_tsv.txt", "r") as f:
    ptb_tsv_content = f.read()


def main():
    parser = argparse.ArgumentParser(description="Complete the corpus with external data.")
    parser.add_argument('-ptb', '--penn-tree-bank', type=str, help='Path to the folder containing all '
                                                                   'POS tagged files in the Penn Tree Bank. '
                                                                   'In version 2.0 of the PTB, this is the folder '
                                                                   '"tagged", in version 3.0 it is "tagged/pos".',
                        required=False)
    parser.add_argument('-amr', '--amr-corpus', type=str, help='Path to the concatenated AMR 3.0 testset file. To '
                                                               'concatenate the AMR testset files, you can use '
                                                               'concatenate_amr_files.py.',
                        required=False)
    args = parser.parse_args()

    if args.penn_tree_bank is not None:

        print("Adding Penn Treebank data to GrAPES corpus...")

        ptb_filepath = args.penn_tree_bank + "/wsj/00/wsj_0060.pos"

        key = make_key_from_file_contents(ptb_filepath)

        fernet = Fernet(key)

        decrypted_ptb_txt = fernet.decrypt(ptb_txt_content.encode('utf-8')).decode('utf-8')
        decrypted_ptb_tsv = fernet.decrypt(ptb_tsv_content.encode('utf-8')).decode('utf-8')

        write_ptb_data(decrypted_ptb_txt, decrypted_ptb_tsv)

    if args.amr_corpus is not None:

        print("\nAdding Word Disambiguation (handcrafted) data from AMR corpus to GrAPES corpus...")
        update_from_amr_testset(args.amr_corpus)

    if args.amr_corpus is None and args.penn_tree_bank is None:
        print("One of the optional arguments is required!\n")
        parser.print_help()


def make_key_from_file_contents(file_name):
    full_string = ""
    with open(file_name, "r") as f:
        for line in f.readlines():
            line = line.strip().replace(" ", "")
            full_string += line
    # print(full_string)
    otherinfo = b"concatkdf-example"
    ckdf = ConcatKDFHash(
        algorithm=hashes.SHA256(),
        length=32,
        otherinfo=otherinfo,
    )
    key = ckdf.derive(full_string.encode('utf-8'))
    key = base64.urlsafe_b64encode(key)
    return key


if __name__ == "__main__":
    main()
