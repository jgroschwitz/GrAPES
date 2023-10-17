import os
from xml.etree import ElementTree


def main():
    frames_path = "../../../data/Edinburgh/amr3.0/data/frames/propbank-amr-frames-xml-2018-01-25/"
    sentences_path = "../../../data/new_sentences_raw.txt"
    frame_to_definition_text = dict()
    for filename in os.listdir(frames_path):
        if filename.endswith(".xml"):
            tree = ElementTree.parse(frames_path + filename)
            root = tree.getroot()
            frames_here = set()
            definition_text = ""
            for predicate in root.findall("predicate"):
                for roleset in predicate:
                    if "id" in roleset.attrib and "name" in roleset.attrib:
                        # print(roleset.tag, roleset.attrib)
                        frame_name = roleset.attrib["id"].replace(".", "-").replace("_", "-")
                        frames_here.add(frame_name)
                        definition_text += frame_name + ": " + roleset.attrib["name"] + "\n"
                        roles = roleset.find("roles")
                        if roles:
                            for role in roles:
                                if "n" in role.attrib.keys() and "descr" in role.attrib.keys():
                                    role_label = ":ARG" + role.attrib["n"]
                                    role_def = role.attrib["descr"]
                                    definition_text += "\t" + role_label + ": " + role_def + "\n"
                        definition_text += "\n"
            for frame in frames_here:
                frame_to_definition_text[frame] = definition_text

    with open(sentences_path, "r") as f:
        with open("outputs/sense_prompts.txt", "w") as g:
            frame = None
            word = None
            for i, line in enumerate(f):
                line = line.strip()
                if i % 4 == 0:
                    frame = line.lower()
                    word = frame.split("-")[0]
                elif i % 4 == 1 or i % 4 == 2:
                    g.write("Consider the following definitions:\n\n")
                    g.write(frame_to_definition_text[frame]+"\n")
                    g.write(f"What sense of the word {word} is used in the following sentence?\n\n")
                    g.write(line + "\n\n\n\n\n\n")


if __name__ == "__main__":
    main()
