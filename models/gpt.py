import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    num_layers: int = 12
    num_heads: int = 12
    embed_size: int = 768
    vocab_size: int = 50257
    context_size: int = 1024
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, n, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.embed_size, config.embed_size*3, bias=config.bias)
        self.proj = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, past, past_len):
        B, T, C = x.shape
        H = self.config.num_heads
        q, k, v = self.qkv(x).view(B, T, 3, H, C // H).transpose(1, 3).unbind(dim=2)  # B, H, T, C // H
        if past is not None:  # write KV into the buffer
            assert past_len == 0 or T == 1  # since is_causal has to be disabled
            past[:, :, 0, past_len:past_len+T], past[:, :, 1, past_len:past_len+T] = k, v
            k, v = past[:, :, 0, :past_len+T], past[:, :, 1, :past_len+T]

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=T > 1, dropout_p=self.config.dropout if self.training else 0)
        x = x.transpose(1, 2).contiguous().view(B, T, C)  # join the heads together
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_size, config.embed_size*4, bias=config.bias)
        self.fc2 = nn.Linear(config.embed_size*4, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.gelu(x, approximate='tanh'))
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.ln1 = LayerNorm(config.embed_size, bias=config.bias)
        self.ln2 = LayerNorm(config.embed_size, bias=config.bias)

    def forward(self, x, past, past_len):
        x = x + self.attn(self.ln1(x), past, past_len)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_size)
        self.embed_pos = nn.Embedding(config.context_size, config.embed_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.embed_size, bias=config.bias)
        self.fc_out = nn.Linear(config.embed_size, config.vocab_size, bias=False)

    def forward(self, x, past=None, past_len=0):
        assert past is None or past_len < past.shape[4]
        pos = torch.arange(past_len, past_len+x.shape[-1], dtype=torch.int32, device=x.device)
        x = self.embed_tokens(x) + self.embed_pos(pos)
        x = self.dropout(x)
        for i, block in enumerate(self.blocks):
            x = block(x, past[:,i] if past is not None else None, past_len)
        x = self.fc_out(self.ln(x))
        return x

    @torch.no_grad()
    def generate(self, context, num_tokens=1, temperature=1.0, top_k=-1):
        past = torch.zeros(len(context), self.config.num_layers, self.config.num_heads, 2, self.config.context_size, self.config.embed_size // self.config.num_heads)
        self(context, past, 0)  # init the kv cache
        for i in range(num_tokens-1):
            logits = self(context[:,-1:], past, context.shape[1]+i-1)[:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=-1)
        return context
