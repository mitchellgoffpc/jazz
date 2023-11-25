import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class CLIPConfig:
  num_layers: int = 12
  num_heads: int = 12
  embed_size: int = 768
  vocab_size: int = 49408
  context_size: int = 77

@dataclass
class SDConfig:
  pass


# TEXT ENCODER

class CLIPMLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.fc1 = nn.Linear(config.embed_size, 4*config.embed_size)
    self.fc2 = nn.Linear(4*config.embed_size, config.embed_size)

  def __call__(self, x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x

class CLIPAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.qkv = nn.Linear(config.embed_size, 3*config.embed_size)
    self.proj = nn.Linear(config.embed_size, config.embed_size)

  def __call__(self, x, attn_mask):
    B, T, C = x.shape
    H = self.config.num_heads
    q, k, v = self.qkv(x).view(B, T, 3, H, C // H).transpose(1, 3).unbind(dim=2)  # B, H, T, C // H
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = x.transpose(1, 2).contiguous().view(B, T, C)
    x = self.proj(x)
    return x

class CLIPBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = CLIPAttention(config)
    self.mlp = CLIPMLP(config)
    self.ln1 = nn.LayerNorm(config.embed_size)
    self.ln2 = nn.LayerNorm(config.embed_size)

  def __call__(self, x, attn_mask):
    x = x + self.attn(self.ln1(x), attn_mask)
    x = x + self.mlp(self.ln2(x))
    return x

class CLIPEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_size)
    self.embed_pos = nn.Embedding(config.context_size, config.embed_size)
    self.blocks = nn.ModuleList([CLIPBlock(config) for i in range(config.num_layers)])
    self.ln = nn.LayerNorm(config.embed_size)

  def __call__(self, x):
    attn_mask = torch.full((1, 1, 77, 77), -torch.inf).triu(1)
    pos = torch.arange(x.shape[-1], dtype=torch.int32, device=x.device)
    x = self.embed_tokens(x) + self.embed_pos(pos)
    for block in self.blocks:
      x = block(x, attn_mask)
    return self.ln(x)


# IMAGE ENCODER / DECODER

class ResnetBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.norm1 = nn.GroupNorm(32, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.norm2 = nn.GroupNorm(32, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

  def forward(self, x):
    h = self.conv1(F.silu(self.norm1(x)))
    h = self.conv2(F.silu(self.norm2(h)))
    return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
  def __init__(self, num_channels):
    super().__init__()
    self.norm = nn.GroupNorm(32, num_channels)
    self.qkv = nn.Conv2d(num_channels, 3*num_channels, kernel_size=1)
    self.proj = nn.Conv2d(num_channels, num_channels, kernel_size=1)

  def forward(self, x):
    B, C, H, W = x.shape
    q, k, v = self.qkv(self.norm(x)).view(B, 3, C, H*W).transpose(2, 3).unbind(dim=1)
    h = F.scaled_dot_product_attention(q, k, v)
    h = h.transpose(1, 2).view(B, C, H, W)
    return x + self.proj(h)


class EncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample):
    super().__init__()
    self.conv1 = ResnetBlock(in_channels, out_channels)
    self.conv2 = ResnetBlock(out_channels, out_channels)
    self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if downsample else None

  def forward(self, x):
    x = self.conv2(self.conv1(x))
    if self.downsample:
      x = self.downsample(x)[:,:,:1:,1:]  # pytorch doesn't allow fancy padding
    return x

class DecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, upsample):
    super().__init__()
    self.conv1 = ResnetBlock(in_channels, out_channels)
    self.conv2 = ResnetBlock(out_channels, out_channels)
    self.conv3 = ResnetBlock(out_channels, out_channels)
    self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if upsample else None

  def forward(self, x):
    x = self.conv3(self.conv2(self.conv1(x)))
    if self.upsample:
      B, C, H, W = x.shape
      x = x.view(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, 2*W, 2*H)
      x = self.upsample(x)
    return x



# class Lambda(nn.Module):
#   def __init__(self, f): self.f = f
#   def forward(self, *x): return self.f(*x)
#
# def expand(x):
#   B, C, H, W = x.shape
#   return x.view(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).view(B, C, 2*W, 2*H)
#
# def EncoderBlock(in_channels, out_channels, downsample):
#   return nn.Sequential(OrderedDict({
#     'conv1':  ResnetBlock(in_channels, out_channels),
#     'conv2': ResnetBlock(out_channels, out_channels),
#     'downsample': nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=(0,1,0,1)) if downsample else nn.Identity()}))
#
# def DecoderBlock(in_channels, out_channels, upsample):
#   return nn.Sequential(OrderedDict({
#     'conv1': ResnetBlock(in_channels, out_channels),
#     'conv2': ResnetBlock(out_channels, out_channels),
#     'conv3': ResnetBlock(out_channels, out_channels),
#     'expand': Lambda(expand) if upsample else nn.Identity(),
#     'upsample': nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if upsample else nn.Identity()}))
#
#
#
# class Upsample(nn.Module):
#   def __init__(self, num_channels):
#     super().__init__()
#     self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
#
#   def forward(self, x):
#     B, C, H, W = x.shape
#     x = x.view(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).view(B, C, 2*W, 2*H)
#     return self.conv(x)
#
# def EncoderBlock(in_channels, out_channels, downsample):
#   return nn.Sequential(OrderedDict({
#     'conv1':  ResnetBlock(in_channels, out_channels),
#     'conv2': ResnetBlock(out_channels, out_channels),
#     'downsample': nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=(0,1,0,1)) if downsample else nn.Identity()}))
#
# def DecoderBlock(in_channels, out_channels, upsample):
#   return nn.Sequential(OrderedDict({
#     'conv1': ResnetBlock(in_channels, out_channels),
#     'conv2': ResnetBlock(out_channels, out_channels),
#     'conv3': ResnetBlock(out_channels, out_channels),
#     'upsample': Upsample(out_channels) if upsample else nn.Identity()}))



class Encoder(nn.Module):
  def __init__(self, img_channels, z_channels):
    super().__init__()
    params = [(128, 128, True), (128, 256, True), (256, 512, True), (512, 512, False)]
    in_channels, _, _ = params[0]
    _, out_channels, _ = params[-1]
    self.conv_in = nn.Conv2d(img_channels, in_channels, kernel_size=3, stride=1, padding=1)
    self.blocks = nn.ModuleList([EncoderBlock(*p) for p in params])
    self.conv1 = ResnetBlock(out_channels, out_channels)
    self.attn = AttnBlock(out_channels)
    self.conv2 = ResnetBlock(out_channels, out_channels)
    self.norm = nn.GroupNorm(32, out_channels)
    self.conv_out = nn.Conv2d(out_channels, 2*z_channels, kernel_size=3, padding=1)
    self.conv_quant = nn.Conv2d(2*z_channels, 2*z_channels, kernel_size=1)

  def forward(self, x):
    x = self.conv_in(x)
    for block in self.blocks:
      x = block(x)
    x = self.conv1(x)
    x = self.conv2(self.attn(x))
    x = self.conv_out(F.silu(self.norm(x)))
    x = self.conv_quant(x)
    return x

class Decoder(nn.Module):
    def __init__(self, z_channels, img_channels):
      super().__init__()
      params = [(512, 512, True), (512, 512, True), (512, 256, True), (256, 128, False)]
      in_channels, _, _ = params[0]
      _, out_channels, _ = params[-1]
      self.conv_dequant = nn.Conv2d(z_channels, z_channels, kernel_size=1)
      self.conv_in = nn.Conv2d(z_channels, in_channels, kernel_size=3, padding=1)
      self.conv1 = ResnetBlock(in_channels, in_channels)
      self.attn = AttnBlock(in_channels)
      self.conv2 = ResnetBlock(in_channels, in_channels)
      self.blocks = nn.ModuleList([DecoderBlock(*p) for p in params])
      self.norm = nn.GroupNorm(32, out_channels)
      self.conv_out = nn.Conv2d(out_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x):
      x = self.conv_dequant(x)
      x = self.conv_in(x)
      x = self.conv1(x)
      x = self.conv2(self.attn(x))
      for block in self.blocks:
        x = block(x)
      x = self.conv_out(F.silu(self.norm(x)))
      return x
