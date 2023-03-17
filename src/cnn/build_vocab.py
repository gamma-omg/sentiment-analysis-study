import os
import argparse
import spacy
import csv
from tqdm import tqdm
from utils import Vocabulary, Tokenizer


def build_vocab(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    nlp = spacy.load("en_core_web_lg")
    tokenizer = Tokenizer(nlp)
    vocab = Vocabulary(nlp)

    with open(args.data, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        rows_count = sum(1 for _ in reader) - 1
        f.seek(0)

        next(reader)

        for row in tqdm(reader, desc="Building vocab", total=rows_count):
            vocab.update(tokenizer.tokenize(row[0]))

    vocab.truncate(args.max_size)
    vocab.save(os.path.join(args.out, "vocab.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--max_size", type=int, default=20000)
    parser.add_argument("--out", type=str, required=True)
    build_vocab(parser.parse_args())
    