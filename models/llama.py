import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    embed_size: int = 4096
    intermediate_size: int = 14336
    vocab_size: int = 128256
    context_size: int = 8192
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    bias: bool = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * freqs_cis[None, None]).flatten(start_dim=3)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm + self.eps)
        return self.weight * x_normed

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_size = config.embed_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_size // self.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.embed_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.embed_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.embed_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, past, past_len):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if past is not None:
            assert past_len == 0 or T == 1
            past[:, :, 0, past_len:past_len+T], past[:, :, 1, past_len:past_len+T] = k, v
            k, v = past[:, :, 0, :past_len+T], past[:, :, 1, :past_len+T]

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=T > 1, dropout_p=self.config.dropout if self.training else 0)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.o_proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.embed_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.embed_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))

class Block(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ff = FeedForward(config)
        self.ln1 = RMSNorm(config.embed_size, eps=config.rms_norm_eps)
        self.ln2 = RMSNorm(config.embed_size, eps=config.rms_norm_eps)

    def forward(self, x, freqs_cis, past, past_len):
        x = x + self.attn(self.ln1(x), freqs_cis, past, past_len)
        x = x + self.ff(self.ln2(x))
        return x

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln = RMSNorm(config.embed_size, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(config.embed_size, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.config.embed_size // self.config.num_heads,
            self.config.context_size * 2
        )

    def forward(self, x, past=None, past_len=0):
        assert past is None or past_len < past.shape[4]
        _, T = x.shape
        freqs_cis = self.freqs_cis[past_len:past_len+T]

        x = self.embed_tokens(x)
        for i, block in enumerate(self.blocks):
            x = block(x, freqs_cis, past[:,i] if past is not None else None, past_len)
        x = self.out_proj(self.ln(x))
        return x

    @torch.no_grad()
    def generate(self, context, num_tokens=1, temperature=1.0, top_k=-1):
        past = torch.zeros(len(context), self.config.num_layers, self.config.num_kv_heads, 2, self.config.context_size, self.config.embed_size // self.config.num_heads)
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
