#!/usr/bin/env python
import sys
import json
import struct
import torch
import requests
import tiktoken
from pathlib import Path
from tqdm import tqdm, trange
from model import CLIPConfig, SDConfig, CLIPEncoder, Encoder, Decoder

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


def fix_state_dict(state_dict):
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
  state_dict.pop('position_ids')
  return state_dict


CONFIGS = {
  'v1.4': SDConfig(),
  'v1.5': SDConfig()}

REPOS = {
  'v1.4': 'CompVis/stable-diffusion-v1-4',
  'v1.5': 'runwayml/stable-diffusion-v1-5'}

COMPONENTS = {
  'text_encoder': 'text_encoder/model',
  # 'vae': 'vae/diffusion_pytorch_model',
  # 'unet': 'unet/diffusion_pytorch_model'
}

if __name__ == '__main__':
  device = 'cpu'
  model = sys.argv[1]
  assert model in CONFIGS

  # Load weights
  weights_urls = {k: f'https://huggingface.co/{REPOS[model]}/resolve/main/{path}.safetensors' for k,path in COMPONENTS.items()}
  checkpoint_fns = {k: Path(f'/tmp/stable-diffusion/{model}/{k}.safetensors') for k in COMPONENTS}
  state_dict = load_checkpoint(weights_urls, checkpoint_fns)
  for k in state_dict:
    state_dict[k] = fix_state_dict(state_dict[k])

  # Load text encoder
  config = CLIPConfig()
  text_encoder = CLIPEncoder(config).to(device)
  text_encoder.load_state_dict(state_dict['text_encoder'])

  context = torch.randint(0, config.vocab_size, size=(1, config.context_size)).to(device)
  with torch.no_grad():
    data = text_encoder(context)

  # Load image encoder/decoder
  encoder = Encoder(3, 4).to(device)
  decoder = Decoder(4, 3).to(device)

  img = torch.randint(0, 255, size=(1, 3, 256, 256)).float().to(device)
  with torch.no_grad():
    z = encoder(img)
    pred = decoder(z[:,:4])  # mean only
