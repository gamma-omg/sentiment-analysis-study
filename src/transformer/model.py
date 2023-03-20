import torch
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        
        encoding = torch.zeros(max_len, embed_dim)

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, embed_dim, step=2).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_dim)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_dim)))
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        seq_len = x.shape[-1]
        return self.encoding[:seq_len, :]    


class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x ~ [B, T, C]
        q = self.query(x) # [B, T, C]
        k = self.key(x) # [B, T, C]
        v = self.value(x) # [B, T, C]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim) # [B, T, T]
        attn = torch.softmax(attn, dim=-1) # [B, T, T]
        attn = attn @ v         
        return attn # [B, T, C]
    

# class MutiHeadedAttention(torch.nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.heads = torch.nn.ModuleList([AttentionHead(embed_dim) for _ in range(num_heads)])
#         self.linear = torch.nn.Linear(embed_dim * num_heads, embed_dim)

#     def forward(self, x):
#         # x ~ B, T, C
#         y = torch.cat([head(x) for head in self.heads], dim=2) # [B, T, C * num_heads]
#         return self.linear(y)

class ScaleDotProductAttention(torch.nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range        
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_concat = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class EncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.5):
        super().__init__()
        self.attn = MultiHeadedAttention(embed_dim, num_heads)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, embed_dim)
        )        

    def forward(self, x, mask=None):
        attn = self.attn(q=x, k=x, v=x, mask=mask)
        attn = self.dropout(attn)
        norm = self.norm1(attn + x)
        ff = self.ff(norm)
        return self.norm2(ff + norm)
    

class TransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, hidden_dim, num_heads, num_layers, unk_index, pad_index, dropout=0.1):
        super().__init__()

        self.unk_index = unk_index
        self.pad_index = pad_index
        self.max_length = max_len

        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.data[unk_index] = torch.zeros(embed_dim)
        self.embed.weight.data[pad_index] = torch.zeros(embed_dim)

        self.pos = PositionalEncoding(max_len, embed_dim)
        self.encoder = torch.nn.ModuleList([EncoderBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.ff_c = torch.nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = x[:, :self.max_length]
        e = self.embed(x)

        p = self.pos(x)
        y = e + p

        pad_mask = self.make_pad_mask(x, x, self.pad_index, self.pad_index)        
        for layer in self.encoder:
            y = layer(y, mask=pad_mask)

        y = y.mean(dim=1)
        y = self.ff_c(y)
        return y
    
    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask