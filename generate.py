#!/usr/bin/env python
import os
import json
import struct
import torch
import argparse
import requests
import tiktoken
import safetensors
from pathlib import Path
from tqdm import tqdm, trange
from models.gpt import GPT, GPTConfig
from models.llama import Llama, LlamaConfig

CHECKPOINT_DIR = Path(__file__).parent / 'pretrained'
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
GPT2_URL = "https://huggingface.co/{model}/resolve/main/model.safetensors"
LLAMA3_URL = "https://huggingface.co/meta-llama/Meta-Llama-3-{size}/resolve/main/model-{index:05d}-of-00004.safetensors"

CONFIGS = {
    'gpt2': GPTConfig(num_layers=12, num_heads=12, embed_size=768),
    'gpt2-medium': GPTConfig(num_layers=24, num_heads=16, embed_size=1024),
    'gpt2-large': GPTConfig(num_layers=36, num_heads=20, embed_size=1280),
    'gpt2-xl': GPTConfig(num_layers=48, num_heads=25, embed_size=1600),
    'llama3-8b': LlamaConfig(num_layers=32, num_heads=32, embed_size=4096)}


def load_checkpoint(weights_url, checkpoint_fn, dtype=torch.float32):
    tmp_checkpoint_fn = f'{checkpoint_fn}.tmp'
    if not os.path.exists(checkpoint_fn):
        headers = {'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'} if HUGGINGFACE_API_KEY else {}
        r = requests.get(weights_url, headers=headers, stream=True)
        r.raise_for_status()
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
                state_dict[k] = torch.frombuffer(tensor_data[s:e], dtype=dtype).view(v['shape'])
        return state_dict

def load_checkpoints(weights_urls, checkpoint_fns, dtype=torch.float32):
    state_dict = {}
    for weights_url, checkpoint_fns in zip(weights_urls, checkpoint_fns):
        state_dict.update(load_checkpoint(weights_url, checkpoint_fns, dtype=dtype))
    return state_dict


def fix_gpt_state_dict(state_dict):
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

    for src, dst in replacements.items():
        state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
    state_dict = {k:v for k,v in state_dict.items() if not any(x in k for x in biases)}
    state_dict = {k: v.transpose(-1, -2) if any(x in k for x in linears) else v for k,v in state_dict.items()}
    state_dict['fc_out.weight'] = state_dict['embed_tokens.weight']
    return state_dict

def fix_llama_state_dict(state_dict):
    replacements = {
        'model.': '',
        'layers.': 'blocks.',
        'input_layernorm': 'ln1',
        'post_attention_layernorm': 'ln2',
        'self_attn.': 'attn.',
        'mlp.': 'ff.',
        'norm.': 'ln.',
        'lm_head.': 'out_proj.'}
    for src, dst in replacements.items():
        state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
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
    elif args.model.startswith('gpt2'):
        checkpoint_fn = CHECKPOINT_DIR / f'{args.model}.safetensors'
        weights_url = GPT2_URL.format(model=args.model)
        state_dict = load_checkpoint(weights_url, checkpoint_fn)
        state_dict = fix_gpt_state_dict(state_dict)
    elif args.model.startswith('llama3'):
        assert HUGGINGFACE_API_KEY, "HUGGINGFACE_API_KEY must be set to download llama models"
        weights_urls = [LLAMA3_URL.format(size=args.model.removeprefix('llama3-'), index=i) for i in range(1, 5)]
        checkpoint_fns = [CHECKPOINT_DIR / f'{args.model}-{i:05d}.safetensors' for i in range(len(weights_urls))]
        state_dict = load_checkpoints(weights_urls, checkpoint_fns, dtype=torch.bfloat16)
        state_dict = fix_llama_state_dict(state_dict)

    # Create the model
    device = torch.device('cpu')
    config = CONFIGS[args.model]
    model = GPT(config) if isinstance(config, GPTConfig) else Llama(config)
    model = model.to(device).eval()
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
