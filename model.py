import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
  num_layers: int = 12
  num_heads: int = 16
  embed_size: int = 1024
  block_size: int = 1024
  vocab_size: int = 50304
  dropout: float = 0.0
  bias: bool = False


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
    self.qkv = nn.Linear(config.embed_size, config.embed_size*3, bias=config.bias)
    self.proj = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    B, T, C = x.shape
    H = self.config.num_heads
    q, k, v = self.qkv(x).view(B, T, 3, H, C // H).transpose(1, 3).unbind(dim=2)  # B, H, T, C // H
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, dropout_p=self.config.dropout if self.training else 0)
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
    x = self.fc2(F.gelu(x))
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = Attention(config)
    self.mlp = MLP(config)
    self.ln1 = LayerNorm(config.embed_size, bias=config.bias)
    self.ln2 = LayerNorm(config.embed_size, bias=config.bias)

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_size)
    self.embed_pos = nn.Embedding(config.block_size, config.embed_size)
    self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
    self.dropout = nn.Dropout(config.dropout)
    self.ln = LayerNorm(config.embed_size, bias=config.bias)
    self.fc_out = nn.Linear(config.embed_size, config.vocab_size)

  def forward(self, x):
    x = self.embed_tokens(x) + self.embed_pos(x)
    x = self.dropout(x)
    for block in self.blocks:
      x = block(x)
    x = self.fc_out(self.ln(x))
    return x
