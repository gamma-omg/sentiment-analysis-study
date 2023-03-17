import torch


class FastTextModel (torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, unk_index=0, pad_index=1):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data[unk_index] = torch.zeros(embedding_dim)
        self.embedding.weight.data[pad_index] = torch.zeros(embedding_dim)

        self.linear = torch.nn.Linear(embedding_dim, 1)


    def forward(self, x_in):
        e = self.embedding(x_in)
        n = (e[:,:-1] + e[:,1:])
        y = torch.cat([e, n], dim=1)
        y = torch.mean(y, dim=1)
        y_out = self.linear(y)
        return y_out        
    
