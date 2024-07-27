#!/usr/bin/env python
import os
import sys
import json
import struct
import torch
import argparse
import requests
import tiktoken
import safetensors
from tqdm import tqdm, trange
from models.gpt import GPT, GPTConfig

CONFIGS = {
    'gpt2': GPTConfig(num_layers=12, num_heads=12, embed_size=768),
    'gpt2-medium': GPTConfig(num_layers=24, num_heads=16, embed_size=1024),
    'gpt2-large': GPTConfig(num_layers=36, num_heads=20, embed_size=1280),
    'gpt2-xl': GPTConfig(num_layers=48, num_heads=25, embed_size=1600)}


def load_checkpoint(weights_url, checkpoint_fn):
    tmp_checkpoint_fn = f'{checkpoint_fn}.tmp'
    if not os.path.exists(checkpoint_fn):
        r = requests.get(weights_url, stream=True)
        file_size = int(r.headers['content-length'])
        chunk_size = 128 * 1000  # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
        with open(tmp_checkpoint_fn, 'wb') as f:
            with tqdm(desc="Fetching " + weights_url, total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
        os.rename(tmp_checkpoint_fn, checkpoint_fn)

    with open(checkpoint_fn, 'rb') as f:
        header_len, = struct.unpack('<Q', f.read(8))
        metadata = json.loads(f.read(header_len))
        tensor_data = bytearray(f.read()).copy()
        state_dict = {}
        for k,v in metadata.items():
            if k != '__metadata__':
                s, e = v['data_offsets']
                state_dict[k] = torch.frombuffer(tensor_data[s:e], dtype=torch.float32).view(v['shape'])
        return state_dict


def fix_state_dict(state_dict):
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
        'ln_f.': 'ln.'}
    linears = ['attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']
    biases = ['attn.bias', 'attn.masked_bias']

    for src,dst in replacements.items():
        state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
    state_dict = {k:v for k,v in state_dict.items() if not any(x in k for x in biases)}
    state_dict = {k: v.transpose(-1, -2) if any(x in k for x in linears) else v for k,v in state_dict.items()}
    state_dict['fc_out.weight'] = state_dict['embed_tokens.weight']
    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT model generator')
    parser.add_argument('model', choices=CONFIGS.keys(), help='Model configuration to use')
    parser.add_argument('-f', '--file', help='Path to the checkpoint file to load')
    parser.add_argument('-b', '--benchmark', action='store_true', help='Run a benchmark')
    parser.add_argument('-p', '--prompt', help='Prompt to generate completion for')
    args = parser.parse_args()

    # Load the checkpoint
    if args.file:
        with safetensors.safe_open(args.file, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
    else:
        weights_url = f'https://huggingface.co/{args.model}/resolve/main/model.safetensors'
        checkpoint_fn = f'/tmp/{args.model}.safetensors'
        state_dict = load_checkpoint(weights_url, checkpoint_fn)
        state_dict = fix_state_dict(state_dict)

    # Create the model
    device = torch.device('cpu')
    config = CONFIGS[args.model]
    model = GPT(config).to(device).eval()
    model.load_state_dict(state_dict)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Benchmark
    if args.benchmark:
        large_context = torch.randint(0, 50257, size=(1, 1024)).to(device)
        with torch.no_grad():
            for i in trange(1024, desc="benchmarking bs=1, seqlen=1"):
                data = model(large_context[:,i:i+1])

    # Decode
    else:
        prompt = args.prompt or "From fairest creatures"
        context = torch.tensor(tokenizer.encode(prompt))[None].to(device)
        result = model.generate(context, num_tokens=50, top_k=10)
        print(f"Prompt:    ", prompt)
        print(f"Completion:", tokenizer.decode(result[0].tolist()))