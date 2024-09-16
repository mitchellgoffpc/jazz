import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int | None = None
    embed_size: int = 768
    vocab_size: int = 50257
    context_size: int = 1024
    expansion_ratio: float = 4
    norm_eps: float = 1e-5
    dropout: float = 0.0
    use_bias: bool = True
    tie_weights: bool = False
    use_rms_norm: bool = False
    use_gated_mlp: bool = False
    use_rotary_embeddings: bool = False

@dataclass
class LlamaConfig(GPTConfig):
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    embed_size: int = 4096
    vocab_size: int = 128256
    context_size: int = 8192
    expansion_ratio: float = 3.5
    use_bias: bool = False
    use_rms_norm: bool = True
    use_gated_mlp: bool = True
    use_rotary_embeddings: bool = True


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=getattr(module, 'std', 0.02))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim - 1, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * freqs_cis[None, None]).flatten(start_dim=3).type_as(x)


class LayerNorm(nn.Module):
    def __init__(self, n: int, use_bias: bool, use_rms_norm: bool, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.use_rms_norm = use_rms_norm
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n)) if use_bias else None

    def forward(self, x):
        if self.use_rms_norm:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * self.weight + (self.bias if self.bias is not None else 0)
        else:
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.embed_size // self.num_heads

        self.q = nn.Linear(config.embed_size, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k = nn.Linear(config.embed_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.v = nn.Linear(config.embed_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.out = nn.Linear(config.embed_size, config.embed_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.out.std = 0.02 / math.sqrt(2 * config.num_layers)

    def forward(self, x, past, past_len, freqs_cis=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # B, H, T, C // H
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.config.use_rotary_embeddings:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        if past is not None:  # write KV into the buffer
            assert past_len == 0 or T == 1  # since is_causal has to be disabled
            past[:, :, 0, past_len:past_len+T], past[:, :, 1, past_len:past_len+T] = k, v
            k, v = past[:, :, 0, :past_len+T], past[:, :, 1, :past_len+T]

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=T > 1, dropout_p=self.config.dropout if self.training else 0)
        x = x.transpose(1, 2).contiguous().view(B, T, C)  # join the heads together
        x = self.out(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.embed_size, int(config.embed_size * config.expansion_ratio), bias=config.use_bias) if config.use_gated_mlp else None
        self.up = nn.Linear(config.embed_size, int(config.embed_size * config.expansion_ratio), bias=config.use_bias)
        self.down = nn.Linear(int(config.embed_size * config.expansion_ratio), config.embed_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.down.std = 0.02 / math.sqrt(2 * config.num_layers)

    def forward(self, x):
        if self.gate is None:
            x = F.gelu(self.up(x), approximate='tanh')
        else:
            x = F.silu(self.gate(x)) * self.up(x)
        x = self.down(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.ff = FeedForward(config)
        self.ln1 = LayerNorm(config.embed_size, use_bias=config.use_bias, use_rms_norm=config.use_rms_norm, eps=config.norm_eps)
        self.ln2 = LayerNorm(config.embed_size, use_bias=config.use_bias, use_rms_norm=config.use_rms_norm, eps=config.norm_eps)

    def forward(self, x, past, past_len, freqs_cis=None):
        x = x + self.attn(self.ln1(x), past, past_len, freqs_cis)
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_size)
        if config.use_rotary_embeddings:
            self.register_buffer('freqs_cis', precompute_freqs_cis(self.config.embed_size // self.config.num_heads, self.config.context_size), persistent=False)
        else:
            self.embed_pos = nn.Embedding(config.context_size, config.embed_size)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.embed_size, use_bias=config.use_bias, use_rms_norm=config.use_rms_norm, eps=config.norm_eps)
        self.out = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        if config.tie_weights:
            self.embed_tokens.weight = self.out.weight
        self.apply(init_weights)

    def forward(self, x, past=None, past_len=0):
        assert past is None or past_len < past.shape[4]
        _, T = x.shape
        if self.config.use_rotary_embeddings:
            x = self.embed_tokens(x)
            freqs_cis = self.freqs_cis[past_len:past_len + T]
        else:
            pos = torch.arange(past_len, past_len+T, dtype=torch.int32, device=x.device)
            x = self.embed_tokens(x) + self.embed_pos(pos)
            freqs_cis = None

        x = self.dropout(x)
        for i, block in enumerate(self.blocks):
            x = block(x, past[:,i] if past is not None else None, past_len, freqs_cis)
        x = self.out(self.ln(x))
        return x

    @torch.no_grad
    def generate(self, context, num_tokens=1, temperature=1.0, top_k=-1):
        num_kv_heads = self.config.num_kv_heads or self.config.num_heads
        past = torch.zeros(len(context), self.config.num_layers, num_kv_heads, 2, self.config.context_size, self.config.embed_size // self.config.num_heads, device=context.device)
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
