import os
import torch
import argparse
from model import FastTextModel
from utils import Tokenizer, Vocabulary

import csv

class SentimentPrdictor:
    def __init__(self, model_path, vocab, tokenizer):
        self.model = FastTextModel(len(vocab), 30)
        self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
        self.model.eval()
        
        self.tokenizer = tokenizer
        self.vocab = vocab

    def predict(self, sentence):
        with torch.no_grad():
            tokens = self.tokenizer.tokenize(sentence)
            x = [self.vocab.get_index(t) for t in tokens]
            x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
            y = self.model(x)
            y = torch.sigmoid(y)
            y = y.squeeze().item()
            return y
        
def predict(args):
    vocab = Vocabulary()
    vocab.load(os.path.join(args.model, "vocab.txt"))
    tokenizer = Tokenizer() 
            
    predictor = SentimentPrdictor(args.model, vocab, tokenizer)
    print(predictor.predict(args.prompt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    predict(parser.parse_args())