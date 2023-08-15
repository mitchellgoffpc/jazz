import os
import struct
import torch
import requests
import tiktoken
from tqdm import tqdm, trange
from model import GPT, GPTConfig

if __name__ == '__main__':
  # Load weights
  model = 'gpt2'
  weights_url = f'https://huggingface.co/{model}/resolve/main/pytorch_model.bin'
  checkpoint_fn = f'/tmp/{model}.ckpt'
  tmp_checkpoint_fn = f'{checkpoint_fn}.tmp'

  if not os.path.exists(checkpoint_fn):
    r = requests.get(weights_url, stream=True)
    file_size = int(r.headers['content-length'])
    chunk_size = 1000  # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
    with open(tmp_checkpoint_fn, 'wb') as f:
      with tqdm(ncols=100, desc="Fetching " + weights_url, total=file_size, unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
          pbar.update(chunk_size)
    os.rename(tmp_checkpoint_fn, checkpoint_fn)

  state_dict = torch.load(checkpoint_fn)

  # Remap names
  replacements = {
    'h.': 'blocks.',
    'wte.': 'embed_tokens.',
    'wpe.': 'embed_pos.',
    'attn.c_attn': 'attn.qkv',
    'attn.c_proj': 'attn.proj',
    'mlp.c_fc': 'mlp.fc1',
    'mlp.c_proj': 'mlp.fc2',
    'ln_1.': 'ln1.',
    'ln_2.': 'ln2.',
    'ln_f.': 'ln.',
  }
  linears = ['attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']
  biases = ['attn.bias', 'attn.masked_bias']

  for src,dst in replacements.items():
    state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
  state_dict = {k:v for k,v in state_dict.items() if not any(x in k for x in biases)}
  state_dict = {k: v.transpose(-1, -2) if any(x in k for x in linears) else v for k,v in state_dict.items()}
  state_dict['fc_out.weight'] = state_dict['embed_tokens.weight']

  print("Serializing weights...")

  def serialize_layer(data):
    f.write(struct.pack('<Q', data.nelement()))
    f.write(data.numpy().tobytes())

  def serialize_block(key):
    serialize_layer(torch.stack([state_dict[f'blocks.{i}.{key}.weight'] for i in range(GPTConfig.num_layers)], dim=0))
    serialize_layer(torch.stack([state_dict[f'blocks.{i}.{key}.bias'] for i in range(GPTConfig.num_layers)], dim=0))

  with open('/tmp/weights.bin', 'wb') as f:
    f.write(struct.pack('<Q', len(state_dict)))
    serialize_layer(state_dict['embed_tokens.weight'])
    serialize_layer(state_dict['embed_pos.weight'])
    serialize_block('ln1')
    serialize_block('ln2')
    serialize_block('attn.qkv')
    serialize_block('attn.proj')
    serialize_block('mlp.fc1')
    serialize_block('mlp.fc2')
    serialize_layer(state_dict['ln.weight'])
    serialize_layer(state_dict['ln.bias'])
    serialize_layer(state_dict['fc_out.weight'])

  print("Done serializing")

  device = 'cpu'
  config = GPTConfig()
  model = GPT(config).to(device)
  model.load_state_dict(state_dict)
  tokenizer = tiktoken.get_encoding("gpt2")

  prompt = "The capital of Germany is Berlin. The capital of France is"
  context = torch.tensor(tokenizer.encode(prompt))[None].to(device)
  result = model.generate(context, num_tokens=10, top_k=10)
  print(f"Prompt:    ", prompt)
  print(f"Completion:", tokenizer.decode(result[0].tolist()))

  prompt = "Hi my name is Chris but I have been with Nokia for many years."
  context = torch.tensor(tokenizer.encode(prompt))[None].to(device)
  result = model.generate(context, num_tokens=50, top_k=10)
  print(f"Prompt:    ", prompt)
  print(f"Completion:", tokenizer.decode(result[0].tolist()))

  large_context = torch.randint(0, 50257, size=(1, 1024)).to(device)

  with torch.no_grad():
    for i in trange(1024, desc="benchmarking bs=1, seqlen=1"):
      data = model(large_context[:,i:i+1])
