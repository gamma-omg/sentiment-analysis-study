import torch


class Tokenizer:
    def __init__(self, nlp):
        self.nlp = nlp

    def tokenize(self, sentence):
        return [token.text for token in self.nlp(sentence)]


class Vocabulary:
    def __init__(self, nlp, unk_token="<UNK>", pad_token="<PAD>"):
        self.nlp = nlp
        self.w2i = {}
        self.i2w = {}
        self.freqs = {}
        self.idx = 0
        self.ukn_token = unk_token
        self.pad_token = pad_token

        self.add_token(unk_token)
        self.add_token(pad_token)

    def update(self, tokens):
        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        self.freqs[token] = self.freqs.get(token, 0) + 1
        if token not in self.w2i:
            self.w2i[token] = self.idx
            self.i2w[self.idx] = token
            self.idx += 1

    def truncate(self, max_size):
        if len(self.w2i) < max_size:
            return

        sorted_freqs = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        self.w2i = {}
        self.i2w = {}
        self.idx = 0

        self.add_token(self.ukn_token)
        self.add_token(self.pad_token)

        for t, _ in sorted_freqs[:max_size]:
            self.w2i[t] = self.idx
            self.i2w[self.idx] = t
            self.idx += 1            

    def get_index(self, word):
        return self.w2i.get(word, self.w2i["<UNK>"])
    
    def get_token(self, index):
        return self.i2w.get(index, "<UNK>")
    
    def most_common(self, n):
        return sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_unk_index(self):
        return self.w2i[self.ukn_token]
    
    def get_pad_index(self):
        return self.w2i[self.pad_token]
    
    def save(self, path):
        with open(path, "w") as f:
            for token in self.w2i.keys():
                f.write(token + "\n")

    def load(self, path):
        with open(path, "r") as f:
            for token in f.readlines():
                self.add_token(token.strip())

    def get_vectors(self):
        return torch.FloatTensor([self.nlp.vocab.get_vector(w) for w in self.w2i.keys()])

    def __len__(self):
        return len(self.w2i)