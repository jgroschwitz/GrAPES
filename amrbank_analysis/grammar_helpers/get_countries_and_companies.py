from collections import Counter

from amrbank_analysis.util import load_corpus_from_folder, get_node_by_name, get_name


def main():
    training_corpus = load_corpus_from_folder("../../../../data/Edinburgh/amr3.0/data/amrs/split/training/")
    print(len(training_corpus))

    country_counter = Counter()
    company_counter = Counter()
    person_counter = Counter()
    city_counter = Counter()
    for graph in training_corpus:
        for edge in graph.edges(role=":name"):
            ne_type = get_node_by_name(edge.source, graph).target
            name = get_name(edge.target, graph)
            if ne_type == "country":
                wiki_attrs = graph.attributes(source=edge.source, role=":wiki")
                if len(wiki_attrs) > 0:
                    name = f"{name} ({wiki_attrs[0].target})"
                country_counter[name] += 1
            elif ne_type == "company":
                company_counter[name] += 1
            elif ne_type == "person":
                person_counter[name] += 1
            elif ne_type == "city":
                city_counter[name] += 1

    for name, count in country_counter.most_common(50):
        print(name, count)
    print()

    for name, count in company_counter.most_common(50):
        print(name, count)
    print()

    for name, count in person_counter.most_common(50):
        print(name, count)
    print()

    for name, count in city_counter.most_common(50):
        print(name, count)




if __name__ == '__main__':
    main()
