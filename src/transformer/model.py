import torch

class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.register_buffer("scale", torch.tensor(embed_dim).sqrt())

    def forward(self, x):
        # x ~ [B, T, C]
        q = self.query(x) # [B, T, C]
        k = self.key(x) # [B, T, C]
        v = self.value(x) # [B, T, C]
        attn = q @ k.transpose(-2, -1) / self.scale # [B, T, T]
        attn = torch.softmax(attn, dim=-1) # [B, T, T]
        attn = attn @ v         
        return attn # [B, T, C]
    

class MutiHeadedAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.heads = torch.nn.ModuleList([AttentionHead(embed_dim) for _ in range(num_heads)])
        self.linear = torch.nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x):
        y = torch.cat([head(x) for head in self.heads], dim=2)
        return self.linear(y)
    

class EncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.5):
        super().__init__()
        self.attn = MutiHeadedAttention(embed_dim, num_heads)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim)
        )        

    def forward(self, x):
        attn = self.attn(x)
        attn = self.dropout(attn)
        norm = self.norm1(attn + x)
        ff = self.ff(norm)
        ff = self.dropout(ff)
        return self.norm2(ff + norm)
    

class TransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos = torch.nn.Embedding(max_len, embed_dim)
        self.encoder = torch.nn.Sequential(*[EncoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(embed_dim, 1)

    def forward(self, x):
        e = self.embed(x)
        p = self.pos(torch.arange(x.shape[1], device=x.device)).unsqueeze(0).repeat(x.shape[0], 1, 1)
        y = e + p
        y = self.encoder(y)
        y = self.avg_pool(y.permute(0, 2, 1)).squeeze(-1)
        y = self.classifier(y)
        return torch.nn.functional.gelu(y)