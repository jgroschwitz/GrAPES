import pandas


def annotated_tsv2amr_corpus_file(tsv_filepath,
                                  output_filepath,
                                  sentence_column_label,
                                  id_column_label=None,
                                  graph_column_label=None,
                                  comment_column_label=None,
                                  unbounded=False,
                                  new_tsv=False,
                                  ):
    """
    given a tsv file of annotations, generates an AMR corpus file and tsv files for evaluation
    :param unbounded: if this is unbounded dependencies, we expect certain column labels for composing the different tsvs
    :param new_tsv: bool: if true, write to TSV file as well as corpus file
    :param comment_column_label: str: comment column header in input file
    :param graph_column_label: str: graph column header in input file
    :param id_column_label: str: id column header in input file
    :param sentence_column_label: str: sentence column header in input file
    :param tsv_filepath: path to input TSV file
    :param output_filepath: path to write to, including name, minus suffix
                            we'll use this to make output_filepath.txt and .tsv (if needed)
    """
    filename = tsv_filepath.split("/")[-1]

    # read in TSV
    data = pandas.read_csv(tsv_filepath, sep='\t')

    new_tsv_data_as_list = []

    with open(output_filepath + ".txt", 'a') as corpus_file:
        # get all the components of the input file
        for index, (_, row) in enumerate(data.iterrows()):
            # use ID if given, otherwise make one
            given_id = row[id_column_label] if id_column_label else None
            id_code = generate_id(filename, id=given_id, i=index, unbounded=unbounded)

            # sentence
            sentence = row[sentence_column_label]

            # graph
            if graph_column_label:
                graph = row[graph_column_label]
            else:
                graph = "(r / dummy)"  # use dummy graph if none is given

            # comment
            if comment_column_label and not pandas.isna(row[comment_column_label]):
                comments = row[comment_column_label]
            else:
                comments = ""

            if unbounded:
                try:
                    category = row["category"]
                except:
                    print("no category, but expected one")
                    category = ""
                try:
                    distance = row["distance"]
                except:
                    print("no distance, but expected one")
                    distance = ""
                try:
                    parent = row["source"]
                except:
                    print("no source, but expected one")
                    parent = ""
                try:
                    edge = row["edge"]
                except:
                    print("no edge, but expected one")
                    edge = ""
                try:
                    target = row["target"]
                except:
                    print("no target, but expected one")
                    target = ""

            if unbounded:
                corpus_entry = f"# ::id {id_code}\n# ::snt {sentence}\n# ::cat {category}\n# ::distance {distance}\n{graph}\n\n"
            else:
                corpus_entry = f"# ::id {id_code}\n# ::snt {sentence}\n{graph}\n\n"
            corpus_file.write(corpus_entry)

            if new_tsv:
                try:
                    graph = graph.replace('\n', ' ')
                    graph = graph.replace('\t', '')
                except AttributeError:
                    print("problem with graph")
                    print(id_code)
                    print(sentence)
                    print(graph)
                try:
                    comments = comments.replace('\n', " ")
                    comments = comments.replace('\t', '')
                except AttributeError:
                    print("problem with comments")
                    print(id_code)
                    print(comments)

                if unbounded:
                    new_tsv_data_as_list.append([id_code, sentence, parent, edge, target, distance, category, comments])
                else:
                    new_tsv_data_as_list.append([id_code, sentence, graph, comments])

    if new_tsv:
        new_tsv_dataframe = pandas.DataFrame(new_tsv_data_as_list)
        # append to existing TSV
        new_tsv_dataframe.to_csv(output_filepath + ".tsv", sep='\t', mode='a', header=False, index=False)


def generate_id(filename, i, id=None, unbounded=False):
    """
    Build an ID.
    :param filename: if we don't have an ID, use the filename and the row number
    :param i: row number
    :param id: provided id
    :param unbounded: unbounded corpus has some entries with the same id, so we build id-i
    :return:
    """
    if unbounded and id is not None:
        return f"{id}-{i}"
    elif id is not None:
        return id
    return f"{filename}-{i}"


def run_script(output_path_prefix, input_file_dict, header_description_start, new_tsv=True, unbounded=False):
    """
    Run the main script, reading in TSV files and writing to TSV and corpus files
    :param unbounded: if True, we have more columns to deal with
    :param new_tsv: Whether to create a TSV file in the process (if false, this just makes the AMR corpus file)
    :param output_path_prefix: path to where to write corpus and tsv,
                                including file names, minus .txt and.tsv
    :param input_file_dict: keys are input TSV file names, values are a list of headings:
                                [sentence, id, graph, comment]
    :param header_description_start: This will start the header on the corpus file
                                        Followed by list of input files.
    """
    files = ""
    for f in input_file_dict:
        files += f + " "
    header = f"# {header_description_start}, built from {files}\n\n"

    # This can probably be left as is
    input_path = "../../corpus/Annotations/"
    if unbounded:
        input_path += "unbounded/"
    output_tsv = output_path_prefix + ".tsv"
    output_corpus = output_path_prefix + ".txt"

    if new_tsv:
        # initialise output TSV
        if unbounded:
            pandas.DataFrame([], columns=[
                "ID", "sentence", "source", "edge", "target", "distance", "category", "comments"
            ]).to_csv(output_tsv, sep="\t", index=False)
        else:
            # pandas.DataFrame([], columns=["ID", "comments"]).to_csv(output_tsv, sep="\t", index=False)
            pandas.DataFrame([], columns=["ID", "sentence", "graph", "comments"]).to_csv(output_tsv, sep="\t", index=False)

    # initialise output corpus
    with open(output_corpus, 'w') as corpus:
        corpus.write(header)

    # each file has a list of column labels, in this order
    for filename in input_file_dict:
        print("processing", filename)
        annotated_tsv2amr_corpus_file(input_path + filename,
                                      output_path_prefix,
                                      sentence_column_label=input_file_dict[filename][0],
                                      id_column_label=input_file_dict[filename][1],
                                      graph_column_label=input_file_dict[filename][2],
                                      comment_column_label=input_file_dict[filename][3],
                                      unbounded=unbounded,
                                      new_tsv=new_tsv)


if __name__ == "__main__":

    # BERT's Mouth
    output_path = "../../corpus/berts_mouth"

    input_files = {
        "Maria_Bertsmouth1.tsv": ["sentence", None, "graph", "comment", None, None],
        "chris_annotation_BM_54-95.tsv": ["Sentence", "I", "AMR", "Comment", None, None],
        "anna_bert_fixed.tsv": ["Sentence", None, "Annotation", "Comment", None, None],
    }
    header_description = "Putting Words into BERT's Mouth, annotated by students"

    run_script(output_path, input_files, header_description)

    # Winograd
    # output_path = "../../corpus/winograd"
    #
    # input_files = {"annotations_anna_winograd1-30_reviewed.tsv": ["Sentence", None, "Annotation", None, None, None],
    #                "annotations_anna_winograd91-110.tsv": ["Sentence", None, "Annotation", None, None, None],
    #                "chris_annotation_61_90.tsv": ["Sentence", "ID", "AMR", "Comment", None, None],
    #                "chris_annotation_WG_131-150.tsv": ["Sentence", "ID", "AMR", "Comment", None, None],
    #                "Maria_Winograd1.tsv": ["sentence", None, "graph", "comment", None, None],
    #                "Maria_Winograd2.tsv": ["sentence", None, "graph", "comment", None, None],
    #                }
    # header_description = "Winograd, annotated by students"

    # CCG long distance dependencies
    # output_path = "../../corpus/unbounded_dependencies"
    #
    # # files all use the same headers
    # headers = ["sentence", "ID", None, "comment"]
    # input_file_list = [
    #     "relatives_meaghan.tsv",
    #     "object-free-relatives_chris.tsv",
    #     "object-relative-null_chris.tsv",
    #     "object_wh_questions_chris.tsv",
    #     "right_node_raising_chris.tsv",
    #     "subj_relative_embedded_chris.tsv",
    #     "subj_relatives_chris.tsv",
    # ]
    # input_files = {}
    # for f in input_file_list:
    #     input_files[f] = headers
    #
    # header_description = "CCG unbounded dependencies, annotated by Meaghan and students"
    #
    # run script
    # run_script(output_path, input_files, header_description, new_tsv=True, unbounded=True)


    # import argparse
    #
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('-i', '--input_path', required=True,
    #                     help="path to TSV file")
    #
    # parser.add_argument('-o', '--output_path', required=True,
    #                     help="where to write the output AMR corpus file, including its name")
    #
    # parser.add_argument('-s', '--sentence_column', required=True,
    #                     help="label of the sentence column in the TSV")
    #
    # parser.add_argument('-n', '--id_column', default=None,
    #                     help="label of the ID column in the TSV")
    #
    # parser.add_argument('-g', '--graph_column', default=None,
    #                     help="label of the graph column in the TSV")
    #
    # parser.add_argument('-d', '--header', default=None,
    #                     help="will be printed across the top of the file."
    #                          " Don't need to include # or the filename,"
    #                          " as these are added automatically")
    #
    # args = parser.parse_args()
    #
    # print("creating corpus in", args.output_path)

    # annotated_tsv2amr_corpus_file(args.input_path, args.output_path,
    #                               args.sentence_column, id_column_label=args.id_column,
    #                               graph_column_label=args.graph_column,
    #                               header=args.header
    #                               )

    # example with IDs, no graphs
    # annotated_tsv2amr_corpus_file("../corpus/long_distance_dependencies.tsv",
    #                               "../corpus/long_distance_dependencies.txt",
    #                               sentence_column_label="sentence",
    #                               id_column_label="ID",
    #                               header="Long distance dependencies from CCG corpus")
