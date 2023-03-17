import argparse
import spacy
import csv
import json
from tqdm import tqdm


def tokenize_dataset(args):
    nlp = spacy.load("en_core_web_lg")
    ds = []

    with open(args.data, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        rows_count = sum(1 for _ in reader) - 1
        f.seek(0)

        next(reader)
        for row in tqdm(reader, desc="Tokenizing dataset", total=rows_count):
            tokens = nlp(row[0])
            item = {
                "x": [t.text for t in tokens],
                "y": int(row[1])
            }
            ds.append(item)

    with open(args.out, "w") as f:
        json.dump(ds, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    tokenize_dataset(parser.parse_args())