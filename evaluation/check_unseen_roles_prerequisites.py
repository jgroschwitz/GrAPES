from file_utils import read_edge_tsv, get_graph_for_node_string
from graph_matcher import contains_subgraph_modulo_isomorphy
from penman import load, encode


def main():
    id2labels = read_edge_tsv("../", "unseen_roles_new_sentences.tsv")
    print(id2labels)
    gold_amrs = load("../corpus/unseen_roles_new_sentences.txt")
    predicted_amrs = load("../amrbart-output/unseen_roles_new_sentences.txt")
    successes = 0
    fails = 0
    for gold, pred in zip(gold_amrs, predicted_amrs):
        print(gold.metadata["id"])
        for label in id2labels[gold.metadata["id"]]:
            source = get_graph_for_node_string(label[0])
            target = get_graph_for_node_string(label[2])
            missed_one = False
            for sg in [source, target]:
                if not contains_subgraph_modulo_isomorphy(pred, sg):
                    print("missed prerequisite")
                    print(encode(sg))
                    print(encode(pred))
                    print()
                    missed_one = True
            if missed_one:
                fails += 1
            else:
                successes += 1
    print(f"successes: {successes}")
    print(f"fails: {fails}")



if __name__ == "__main__":
    main()
