import os
import sys


def concatenate_text_files(input_dir, output_file):
    """
    Concatenate all text files in a directory into a single file.

    :param input_dir: directory containing text files
    :param output_file: name of output file
    """
    with open(output_file, 'w', encoding="UTF-8") as outfile:
        for fname in os.listdir(input_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(input_dir, fname), encoding="UTF-8") as infile:
                    for line in infile:
                        outfile.write(line)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    concatenate_text_files(input_dir, output_file)
