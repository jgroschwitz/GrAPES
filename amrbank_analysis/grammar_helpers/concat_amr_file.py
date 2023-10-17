import os


def concatenate(subfolder):
    with open(f"../../external_resources/amrs/split/concatenated/{subfolder}.txt", "w") as f:
        for file in sorted(os.listdir("../../external_resources/amrs/split/" + subfolder)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                with open(f"../../external_resources/amrs/split/{subfolder}/{filename}", "r") as f2:
                    f.write(f2.read())
                    f.write("\n")


def main():
    concatenate("training")
    concatenate("dev")
    concatenate("test")


if __name__ == '__main__':
    main()
