import os
import torch
import argparse
import spacy
from model import CNNModel
from utils import Tokenizer, Vocabulary

import csv

class SentimentPrdictor:
    def __init__(self, model_path, vocab, tokenizer):
        self.model = CNNModel(vocab, embedding_dim=300, num_filters=100, filter_sizes=[2,3,4,5])
        self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
        self.model.eval()
        
        self.tokenizer = tokenizer
        self.vocab = vocab

    def predict(self, sentence):
        with torch.no_grad():
            tokens = self.tokenizer.tokenize(sentence)
            x = [self.vocab.get_index(t) for t in tokens]
            if len(x) < 5:
                x = x + [self.vocab.get_pad_index()] * (5 - len(x))

            x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
            y = self.model(x)
            y = torch.sigmoid(y)
            y = y.squeeze().item()
            return y
        
def predict(args):
    nlp = spacy.load("en_core_web_lg")
    vocab = Vocabulary(nlp)
    vocab.load(args.vocab)
    tokenizer = Tokenizer(nlp) 
            
    predictor = SentimentPrdictor(args.model, vocab, tokenizer)
    print(predictor.predict(args.prompt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    predict(parser.parse_args())