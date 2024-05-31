import os
import sys


def concatenate_text_files(input_dir, output_file):
    """
    Concatenate all text files in a directory into a single file.

    :param input_dir: directory containing text files
    :param output_file: name of output file
    """
    with open(output_file, 'w', encoding="UTF-8") as outfile:
        test_files = sorted(os.listdir(input_dir))
        for fname in test_files:
            if fname.endswith('.txt'):
                with open(os.path.join(input_dir, fname), encoding="UTF-8") as infile:
                    for line in infile:
                        outfile.write(line)


if __name__ == '__main__':
    """
    First argument is the AMRBank testset folder path.  I.e. the folder containing files like 
    "amr-release-3.0-amrs-test-bolt.txt". Usually this is the folder "data/amr/split/test".
    
    Second argument is the filepath to where you want the output file to be generated.
    
    Then this script concatenates all txt files in the given folder, in alphabetical order, into the output file.
    """
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    concatenate_text_files(input_dir, output_file)
