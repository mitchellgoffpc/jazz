import math
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
    self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

  def forward(self, x):
    h = self.conv1(F.silu(self.norm1(x)))
    h = self.conv2(F.silu(self.norm2(h)))
    return self.conv_shortcut(x) + h

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
    self.res1 = ResnetBlock(in_channels, out_channels)
    self.res2 = ResnetBlock(out_channels, out_channels)
    self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if downsample else None

  def forward(self, x):
    x = self.res2(self.res1(x))
    if self.downsample:
      x = self.downsample(x)[:,:,1:,1:]  # pytorch doesn't allow fancy padding
    return x

class DecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, upsample):
    super().__init__()
    self.res1 = ResnetBlock(in_channels, out_channels)
    self.res2 = ResnetBlock(out_channels, out_channels)
    self.res3 = ResnetBlock(out_channels, out_channels)
    self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if upsample else None

  def forward(self, x):
    x = self.res3(self.res2(self.res1(x)))
    if self.upsample:
      B, C, H, W = x.shape
      x = x.view(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, 2*W, 2*H)
      x = self.upsample(x)
    return x


class Encoder(nn.Module):
  def __init__(self, img_channels, z_channels):
    super().__init__()
    params = [(128, 128, True), (128, 256, True), (256, 512, True), (512, 512, False)]
    in_channels, _, _ = params[0]
    _, out_channels, _ = params[-1]
    self.conv_in = nn.Conv2d(img_channels, in_channels, kernel_size=3, stride=1, padding=1)
    self.blocks = nn.ModuleList([EncoderBlock(*p) for p in params])
    self.res1 = ResnetBlock(out_channels, out_channels)
    self.attn = AttnBlock(out_channels)
    self.res2 = ResnetBlock(out_channels, out_channels)
    self.norm = nn.GroupNorm(32, out_channels)
    self.conv_out = nn.Conv2d(out_channels, 2*z_channels, kernel_size=3, padding=1)
    self.conv_quant = nn.Conv2d(2*z_channels, 2*z_channels, kernel_size=1)

  def forward(self, x):
    x = x.permute(0,3,1,2).float() / 255
    x = (x * 2.0) - 1.0
    x = self.conv_in(x)
    for block in self.blocks:
      x = block(x)
    x = self.res1(x)
    x = self.res2(self.attn(x))
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
      self.res1 = ResnetBlock(in_channels, in_channels)
      self.attn = AttnBlock(in_channels)
      self.res2 = ResnetBlock(in_channels, in_channels)
      self.blocks = nn.ModuleList([DecoderBlock(*p) for p in params])
      self.norm = nn.GroupNorm(32, out_channels)
      self.conv_out = nn.Conv2d(out_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x):
      x = self.conv_dequant(x)
      x = self.conv_in(x)
      x = self.res1(x)
      x = self.res2(self.attn(x))
      for block in self.blocks:
        x = block(x)
      x = self.conv_out(F.silu(self.norm(x)))
      x = (x + 1.0) / 2.0
      x = x.permute(0,2,3,1).clip(0,1) * 255
      return x


# DIFFUSION MODEL

class ResBlock(nn.Module):
  def __init__(self, in_channels, embed_channels, out_channels):
    super().__init__()
    self.norm1 = nn.GroupNorm(32, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.fc_embed = nn.Linear(embed_channels, out_channels)
    self.norm2 = nn.GroupNorm(32, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

  def forward(self, x, t_embeddings):
    h = self.conv1(F.silu(self.norm1(x))) + self.fc_embed(t_embeddings)[..., None, None]
    h = self.conv2(F.silu(self.norm2(h)))
    return self.conv_shortcut(x) + h

class CrossAttention(nn.Module):
  def __init__(self, query_dim, context_dim, num_heads, head_size):
    super().__init__()
    self.num_heads = num_heads
    self.head_size = head_size
    self.context_dim = context_dim
    self.q = nn.Linear(query_dim, num_heads*head_size, bias=False)
    self.k = nn.Linear(context_dim, num_heads*head_size, bias=False)
    self.v = nn.Linear(context_dim, num_heads*head_size, bias=False)
    self.proj = nn.Linear(num_heads*head_size, query_dim)

  def forward(self, x, context=None):
    B, T, C = x.shape
    context = x if context is None else context
    q, k, v = self.q(x), self.k(context), self.v(context)
    q, k, v = [y.view(B, -1, self.num_heads, self.head_size).transpose(1, 2) for y in (q,k,v)]
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2).view(B, -1, C)
    return self.proj(x)

class GEGLU(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.proj = nn.Linear(dim_in, dim_out * 2)

  def forward(self, x):
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * F.gelu(gate)

class SpatialAttention(nn.Module):
  def __init__(self, channels, context_size, num_heads, head_size):
    super().__init__()
    self.norm_in = nn.GroupNorm(32, channels)
    self.proj_in = nn.Conv2d(channels, num_heads * head_size, kernel_size=1)
    self.norm1 = nn.LayerNorm(channels)
    self.norm2 = nn.LayerNorm(channels)
    self.norm3 = nn.LayerNorm(channels)
    self.attn1 = CrossAttention(channels, channels, num_heads, head_size)
    self.attn2 = CrossAttention(channels, context_size, num_heads, head_size)
    self.geglu = GEGLU(channels, 4*channels)
    self.fc = nn.Linear(4*channels, channels)
    self.proj = nn.Conv2d(num_heads * head_size, channels, kernel_size=1)

  def forward(self, x, context=None):
    B, C, H, W = x.shape
    h = self.proj_in(self.norm_in(x))
    h = h.view(B, C, H*W).permute(0, 2, 1)
    h = h + self.attn1(self.norm1(h))
    h = h + self.attn2(self.norm2(h), context=context)
    h = h = self.fc(self.geglu(self.norm3(h)))
    h = h.permute(0, 2, 1).view(B, C, H, W)
    return x + self.proj(h)

class Downsample(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

  def forward(self, x):
    return self.conv(x)

class Upsample(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

  def forward(self, x):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, 2*H, 2*W)
    return self.conv(x)

class DownBlock(nn.Module):
  def __init__(self, in_channels, out_channels, attention=True, downsample=False):
    super().__init__()
    self.res1 = ResBlock(in_channels, 1280, out_channels)
    self.attn1 = SpatialAttention(out_channels, 768, 8, out_channels // 8) if attention else None
    self.res2 = ResBlock(out_channels, 1280, out_channels)
    self.attn2 = SpatialAttention(out_channels, 768, 8, out_channels // 8) if attention else None
    self.downsample = Downsample(out_channels) if downsample else None

  def forward(self, x, context, t_embeddings):
    x1 = self.res1(x, t_embeddings)
    x1 = self.attn1(x1, context) if self.attn1 else x1
    x2 = self.res2(x1, t_embeddings)
    x2 = self.attn2(x2, context) if self.attn2 else x2
    if self.downsample:
      return x1, x2, self.downsample(x2)
    else:
      return x1, x2

class MidBlock(nn.Module):
  def __init__(self, num_channels):
    super().__init__()
    self.res1 = ResBlock(num_channels, 1280, num_channels)
    self.attn1 = SpatialAttention(num_channels, 768, 8, 160)
    self.res2 = ResBlock(num_channels, 1280, num_channels)

  def forward(self, x, context, t_embeddings):
    x = self.res1(x, t_embeddings)
    x = self.attn1(x, context)
    x = self.res2(x, t_embeddings)
    return x

class UpBlock(nn.Module):
  def __init__(self, in_channels, out_channels, attention=True, upsample=False):
    super().__init__()
    self.res1 = ResBlock(in_channels[0], 1280, out_channels)
    self.attn1 = SpatialAttention(out_channels, 768, 8, out_channels // 8) if attention else None
    self.res2 = ResBlock(in_channels[1], 1280, out_channels)
    self.attn2 = SpatialAttention(out_channels, 768, 8, out_channels // 8) if attention else None
    self.res3 = ResBlock(in_channels[2], 1280, out_channels)
    self.attn3 = SpatialAttention(out_channels, 768, 8, out_channels // 8) if attention else None
    self.upsample = Upsample(out_channels) if upsample else nn.Identity()

  def forward(self, x, x1, x2, x3, context, t_embeddings):
    x = self.res1(torch.cat([x, x1], 1), t_embeddings)
    x = self.attn1(x, context) if self.attn1 else x
    x = self.res2(torch.cat([x, x2], 1), t_embeddings)
    x = self.attn2(x, context) if self.attn2 else x
    x = self.res3(torch.cat([x, x3], 1), t_embeddings)
    x = self.attn3(x, context) if self.attn3 else x
    return self.upsample(x)

class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
    self.embed_t_fc1 = nn.Linear(320, 1280)
    self.embed_t_fc2 = nn.Linear(1280, 1280)

    self.down_blocks = nn.ModuleList([
      DownBlock(320, 320, downsample=True),
      DownBlock(320, 640, downsample=True),
      DownBlock(640, 1280, downsample=True),
      DownBlock(1280, 1280, attention=False)])
    self.mid_block = MidBlock(1280)
    self.up_blocks = nn.ModuleList([
      UpBlock([2560, 2560, 2560], 1280, upsample=True, attention=False),
      UpBlock([2560, 2560, 1920], 1280, upsample=True),
      UpBlock([1920, 1280, 960], 640, upsample=True),
      UpBlock([960, 640, 640], 320, upsample=False)])

    self.norm_out = nn.GroupNorm(32, 320)
    self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

  def forward(self, x, timesteps, context, dim=320, max_period=10000):
    args = timesteps * (-math.log(max_period) * torch.arange(dim // 2) / (dim // 2)).exp()
    t_embeddings = torch.cat([args.cos(), args.sin()], 0)[None]
    t_embeddings = F.silu(self.embed_t_fc1(t_embeddings))
    t_embeddings = self.embed_t_fc2(t_embeddings)
    x = self.conv_in(x)

    saved_inputs = [x]
    for block in self.down_blocks:
      *_, x = inputs = block(x, context, t_embeddings)
      saved_inputs.extend(inputs)
    x = self.mid_block(x, context, t_embeddings)
    for block in self.up_blocks:
      x = block(x, saved_inputs.pop(), saved_inputs.pop(), saved_inputs.pop(), context, t_embeddings)

    return self.conv_out(F.silu(self.norm_out(x)))


# COMBINED MODEL

class StableDiffusion(nn.Module):
  def __init__(self):
    self.alphas_cumprod = torch.zeros(1000)
    self.text_encoder = CLIPEncoder()
    self.decoder = Decoder()
    self.denoiser = UNet()

  def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
    temperature = 1
    sigma_t = 0
    sqrt_one_minus_at = (1-a_t).sqrt()
    #print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

    x_prev = a_prev.sqrt() * pred_x0 + dir_xt
    return x_prev, pred_x0

  def get_model_output(self, unconditional_context, context, latent, timestep, unconditional_guidance_scale):
    # put into diffuser
    latents = self.model.diffusion_model(latent.expand(2, *latent.shape[1:]), timestep, unconditional_context.cat(context, dim=0))
    unconditional_latent, latent = latents[0:1], latents[1:2]

    e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
    return e_t

  def decode(self, x):
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = x.reshape(3,512,512).permute(1,2,0).clip(0,1)*255
    return x.cast(dtypes.uint8) if Device.DEFAULT != "WEBGPU" else x

  def __call__(self, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
    e_t = self.get_model_output(unconditional_context, context, latent, timestep, guidance)
    x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
    #e_t_next = get_model_output(x_prev)
    #e_t_prime = (e_t + e_t_next) / 2
    #x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
    return x_prev.realize()
