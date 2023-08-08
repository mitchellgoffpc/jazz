import os
import torch
import requests
import tiktoken
from tqdm import tqdm, trange
from model import GPT, GPTConfig

if __name__ == '__main__':
  # Load weights
  weights_url = 'https://huggingface.co/gpt2/resolve/main/pytorch_model.bin'
  checkpoint_fn = '/tmp/gpt2.ckpt'

  if not os.path.exists(checkpoint_fn):
    r = requests.get(weights_url, stream=True)
    file_size = int(r.headers['content-length'])
    chunk_size = 1000  # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
    with open(checkpoint_fn, 'wb') as f:
      with tqdm(ncols=100, desc="Fetching " + weights_url, total=file_size, unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
          pbar.update(chunk_size)

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

  device = 'mps'
  config = GPTConfig()
  model = GPT(config).to(device)
  model.load_state_dict(state_dict)

  tokenizer = tiktoken.get_encoding("gpt2")
  prompt = "The capital of Germany is Berlin. The capital of France is"
  context = torch.tensor(tokenizer.encode(prompt))[None].to(device)

  idx = model.generate(context, num_tokens=2, top_k=10)
  print(f"Prompt:    ", prompt)
  print(f"Completion:", tokenizer.decode(idx[0].tolist()))

  large_context = torch.randint(0, 50257, size=(1, 1024)).to(device)

  with torch.no_grad():
    for i in trange(1024, desc="benchmarking bs=1, seqlen=1"):
      data = model(large_context[:,i:i+1])
