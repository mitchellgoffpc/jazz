#!/usr/bin/env python
import sys
import json
import struct
import torch
import requests
import tiktoken
from pathlib import Path
from tqdm import tqdm, trange
from model import CLIPConfig, SDConfig, CLIPEncoder, Encoder, Decoder, UNet
from tokenizer import CLIPTokenizer

def load_checkpoint(weights_urls, checkpoint_fns):
  state_dict = {}

  for component, weights_url in weights_urls.items():
    checkpoint_fn = checkpoint_fns[component]
    tmp_checkpoint_fn = Path(f'{checkpoint_fn}.tmp')
    checkpoint_fn.parent.mkdir(exist_ok=True, parents=True)
    if not checkpoint_fn.exists():
      r = requests.get(weights_url, stream=True)
      r.raise_for_status()
      file_size = int(r.headers['content-length'])
      chunk_size = 1000  # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
      with open(tmp_checkpoint_fn, 'wb') as f:
        with tqdm(desc="Fetching " + weights_url, total=file_size, unit_scale=True) as pbar:
          for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(chunk_size)
      tmp_checkpoint_fn.rename(checkpoint_fn)

    with open(checkpoint_fn, 'rb') as f:
      header_len, = struct.unpack('<Q', f.read(8))
      metadata = json.loads(f.read(header_len))
      tensor_data = bytearray(f.read()).copy()
      state_dict[component] = {}
      for k,v in metadata.items():
        if k != '__metadata__':
          s, e = v['data_offsets']
          if k == 'text_model.embeddings.position_ids':
            e = s + (e - s) // 2  # position_ids is weird
          state_dict[component][k] = torch.frombuffer(tensor_data[s:e], dtype=torch.float32).view(v['shape'])

  return state_dict


def fix_clip_state_dict(state_dict):
  replacements = {
    'text_model.': '',
    'embeddings.': '',
    'encoder.': '',
    'token_embedding.': 'embed_tokens.',
    'position_embedding.': 'embed_pos.',
    'final_layer_norm.': 'ln.',
    'layers.': 'blocks.',
    'self_attn.': 'attn.',
    'layer_norm1.': 'ln1.',
    'layer_norm2.': 'ln2.',
    'out_proj.': 'proj.',
  }
  for src, dst in replacements.items():
    state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
  qkv_keys = [k for k in state_dict if '.k_proj.' in k]
  for k in qkv_keys:
    qkv = [state_dict.pop(k.replace('.k_proj.', f'.{x}_proj.')) for x in 'qkv']
    state_dict[k.replace('.k_proj.', '.qkv.')] = torch.cat(qkv, dim=0)
  state_dict.pop('position_ids', None)
  return state_dict


def fix_vae_state_dict(state_dict):
  replacements = {
    'mid_block.': '',
    'down_blocks.': 'blocks.',
    'up_blocks.': 'blocks.',
    'resnets.0.': 'res1.',
    'resnets.1.': 'res2.',
    'resnets.2.': 'res3.',
    'attentions.0.': 'attn.',
    'downsamplers.0.conv.': 'downsample.',
    'upsamplers.0.conv.': 'upsample.',
    'group_norm.': 'norm.',
    'proj_attn.': 'proj.',
    'conv_norm_out.': 'norm.',
    'post_quant_conv.': 'decoder.conv_dequant.',
    'quant_conv.': 'encoder.conv_quant.',
  }
  for src, dst in replacements.items():
    state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
  qkv_keys = [k for k in state_dict if '.key.' in k]
  for k in qkv_keys:
    qkv = [state_dict.pop(k.replace('.key.', f'.{x}.')) for x in ('query', 'key', 'value')]
    state_dict[k.replace('.key.', '.qkv.')] = torch.cat(qkv, dim=0)
  for k in state_dict:
    if 'qkv.weight' in k or 'proj.weight' in k:
      state_dict[k] = state_dict[k][..., None, None]
  state_dicts = {}
  for c in ('encoder', 'decoder'):
    state_dicts[c] = {k.removeprefix(f'{c}.'): v for k, v in state_dict.items() if k.startswith(c)}
  return state_dicts


def fix_unet_state_dict(state_dict):
  replacements = {
    'time_embedding.linear_1.': 'embed_t_fc1.',
    'time_embedding.linear_2.': 'embed_t_fc2.',
    'conv_norm_out.': 'norm_out.',
    'resnets.0.': 'res1.',
    'resnets.1.': 'res2.',
    'resnets.2.': 'res3.',
    'attentions.0.': 'attn1.',
    'attentions.1.': 'attn2.',
    'attentions.2.': 'attn3.',
    'downsamplers.0.': 'downsample.',
    'upsamplers.0.': 'upsample.',
    'transformer_blocks.0.': '',
    'to_q.': 'q.',
    'to_k.': 'k.',
    'to_v.': 'v.',
    'to_out.0.': 'proj.',
    'proj_out.': 'proj.',
    'ff.net.0.': 'geglu.',
    'ff.net.2.': 'fc.',
    'time_emb_proj.': 'fc_embed.',
    'attn1.norm.': 'attn1.norm_in.',
    'attn2.norm.': 'attn2.norm_in.',
    'attn3.norm.': 'attn3.norm_in.',
  }
  for src, dst in replacements.items():
    state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
  return state_dict


CONFIGS = {
  'v1.4': SDConfig(),
  'v1.5': SDConfig()}

REPOS = {
  'v1.4': 'CompVis/stable-diffusion-v1-4',
  'v1.5': 'runwayml/stable-diffusion-v1-5'}

COMPONENTS = {
  'text_encoder': 'text_encoder/model',
  'vae': 'vae/diffusion_pytorch_model',
  'unet': 'unet/diffusion_pytorch_model'}

if __name__ == '__main__':
  device = 'cpu'
  model = sys.argv[1]
  assert model in CONFIGS

  # Load weights
  weights_urls = {k: f'https://huggingface.co/{REPOS[model]}/resolve/main/{path}.safetensors' for k,path in COMPONENTS.items()}
  checkpoint_fns = {k: Path(f'/tmp/stable-diffusion/{model}/{k}.safetensors') for k in COMPONENTS}
  state_dict = load_checkpoint(weights_urls, checkpoint_fns)

  # Load text encoder
  config = CLIPConfig()
  text_encoder = CLIPEncoder(config).to(device)
  text_encoder.load_state_dict(fix_clip_state_dict(state_dict['text_encoder']))

  context = CLIPTokenizer().encode('A cat lying on the grass')
  context = torch.tensor(context, dtype=torch.int32).to(device)[None]
  with torch.no_grad():
    text_features = text_encoder(context)

  # Load image encoder/decoder
  encoder = Encoder(3, 4).to(device)
  decoder = Decoder(4, 3).to(device)

  vae_state_dict = fix_vae_state_dict(state_dict['vae'])
  encoder.load_state_dict(vae_state_dict['encoder'])
  decoder.load_state_dict(vae_state_dict['decoder'])

  """
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt
  from PIL import Image
  img = Image.open('/Users/mitchell/Downloads/dog.jpg')
  img = cv2.resize(np.array(img), (256, 256))
  img_torch = torch.as_tensor(img[None]).to(device)

  with torch.no_grad():
    z = encoder(img_torch)
    pred = decoder(z[:,:4])  # mean only

  pred = pred.cpu().numpy().astype(np.uint8)[0]
  _, ax = plt.subplots(1, 2, figsize=(8, 4))
  ax[0].imshow(img)
  ax[1].imshow(pred)
  plt.show()
  """

  # Load diffusion model
  model = UNet().to(device)
  model.load_state_dict(fix_unet_state_dict(state_dict['unet']))

  import math
  steps = 50
  latent = torch.randn(1, 4, 64, 64)
  timesteps = torch.tensor(range(1, 1000, 1000//steps), dtype=torch.int32)
  alphas = torch.ones(len(timesteps))
  alphas_prev = torch.cat([torch.tensor([1.0]), alphas[:-1]])

  with torch.no_grad():
    for index, timestep in tqdm(list(enumerate(timesteps))[::-1]):
      e_t = model(latent, timestep, text_features)

      x = latent
      a_t = alphas[index]
      a_prev = alphas_prev[index]
      temperature = 1
      sigma_t = 0
      sqrt_one_minus_at = (1-a_t).sqrt()

      pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

      # direction pointing to x_t
      dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

      x_prev = a_prev.sqrt() * pred_x0 + dir_xt
      latent = x_prev
      print(latent.mean())

  import numpy as np
  import matplotlib.pyplot as plt
  with torch.no_grad():
    pred = decoder(latent)
  pred = pred.cpu().numpy().astype(np.uint8)[0]
  plt.imshow(pred)
  plt.show()
