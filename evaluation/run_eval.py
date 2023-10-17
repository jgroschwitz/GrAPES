from penman import load
from pp_attachment import evaluate_with_seeing_graphs
from long_lists import evaluate_long_lists
import prettytable
from util import num_to_score


def main():
    parsers = [("AM-parser", "../amparser-output/"), ("AMRBART", "../amrbart-output/")]
    phenomena_file_names = ["pp_attachment", "long_lists_short"]
    golds = [load(f"../corpus/{file_name}.txt") for file_name in phenomena_file_names]

    results_table = prettytable.PrettyTable()
    results_table.add_column("Metric", ["PP-attachment",
                                        "  Requirements",
                                        "  UAS",
                                        "  LAS",
                                        "Long Lists",
                                        "  :opi F1",
                                        "  Conjunct F1"])
    results_table.align = "l"

    for parser in parsers:
        folder = parser[1]
        parser_name = parser[0]
        print("Folder: " + folder)
        predictions = [load(folder + file_name+".txt") for file_name in phenomena_file_names]

        pp_req, pp_uas, pp_las = evaluate_with_seeing_graphs(predictions[0], golds[0])

        op_results, conjunct_results = evaluate_long_lists(predictions[1], golds[1])

        results_table.add_column(parser_name,
                                 ["",
                                     num_to_score(pp_req),
                                     num_to_score(pp_uas),
                                     num_to_score(pp_las),
                                     "",
                                     num_to_score(op_results[2]),
                                     num_to_score(conjunct_results[2])])

    print(results_table)




if __name__ == "__main__":
    main()
