import torch
import torch.nn.functional as F
import spacy

class CNNModel (torch.nn.Module):
    def __init__(self, vocab, embedding_dim, num_filters, filter_sizes, dropout=0.5):
        super().__init__()
                
        self.embeddings = torch.nn.Embedding(len(vocab), embedding_dim)        
        self.embeddings.weight.data.copy_(vocab.get_vectors())
        self.embeddings.weight.requires_grad = False

        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x_in):
        e = self.embeddings(x_in).unsqueeze(3)
        c = [F.relu(conv(e.permute(0, 3, 1, 2)).squeeze(3)) for conv in self.convs]
        m = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in c]
        t = torch.cat(m, dim=1)
        t = self.dropout(t)
        return self.linear(t)