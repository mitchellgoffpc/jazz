import os
import types
import torch
import requests
import tiktoken
import safetensors
from tqdm import tqdm
from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from tiktoken.load import load_tiktoken_bpe
from models.gpt import GPT, GPTConfig, LlamaConfig

Tokenizer = Any
StateDict = dict[str, torch.Tensor]

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
CHECKPOINT_DIR = Path(__file__).parent / 'pretrained'

GPT2_URL = "https://huggingface.co/{model}/resolve/main/model.safetensors"
LLAMA3_URL = "https://huggingface.co/meta-llama/Meta-Llama-3-{size}/resolve/main/model-{index:05d}-of-00004.safetensors"
LLAMA3_TOKENIZER_URL = 'https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/original/tokenizer.model'
LLAMA3_MODEL_PATH = CHECKPOINT_DIR / 'llama-tokenizer.model'

CONFIGS = {
    'gpt2': GPTConfig(num_layers=12, num_heads=12, embed_size=768),
    'gpt2-medium': GPTConfig(num_layers=24, num_heads=16, embed_size=1024),
    'gpt2-large': GPTConfig(num_layers=36, num_heads=20, embed_size=1280),
    'gpt2-xl': GPTConfig(num_layers=48, num_heads=25, embed_size=1600),
    'llama3-8b': LlamaConfig(num_layers=32, num_heads=32, embed_size=4096)}


def download_file(url: str, path: Path, chunk_size: int = 128000, headers: dict[str, str] = {}) -> None:
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    tmp_path = Path(f'{path}.tmp')
    file_size = int(r.headers.get('content-length', 0))
    with open(tmp_path, 'wb') as f, tqdm(desc="Fetching " + url, total=file_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(chunk_size)
    tmp_path.rename(path)

def load_checkpoint(weights_url: str, checkpoint_path: Path) -> StateDict:
    download_file(weights_url, checkpoint_path, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})
    with safetensors.safe_open(checkpoint_path, framework='pt') as f:
        return {k: f.get_tensor(k) for k in f.keys()}

def load_checkpoints(weights_urls: list[str], checkpoint_paths: list[Path]) -> StateDict:
    state_dict = {}
    for weights_url, checkpoint_path in zip(weights_urls, checkpoint_paths):
        state_dict.update(load_checkpoint(weights_url, checkpoint_path))
    return state_dict


# Tokenizers

def load_gpt2_tokenizer() -> Tokenizer:
    return tiktoken.get_encoding('gpt2')

def load_llama3_tokenizer() -> Tokenizer:
    download_file(LLAMA3_TOKENIZER_URL, LLAMA3_MODEL_PATH, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})
    mergeable_ranks = load_tiktoken_bpe(str(LLAMA3_MODEL_PATH))
    num_base_tokens = len(mergeable_ranks)
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, num_reserved_special_tokens - 5)
    ]
    special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
    tokenizer = tiktoken.Encoding(
        name=LLAMA3_MODEL_PATH.name,
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    def encode(text: str): return [tokenizer._special_tokens['<|begin_of_text|>'], *tokenizer.encode(text)]
    def decode(tokens: str): return tokenizer.decode(tokens[1:])  # remove begin_of_text token
    return types.SimpleNamespace(encode=encode, decode=decode)

def load_pretrained_tokenizer(checkpoint: str) -> Tokenizer:
    if checkpoint.startswith('gpt2'):
        return load_gpt2_tokenizer()
    elif checkpoint.startswith('llama3'):
        return load_llama3_tokenizer()
    else:
        raise RuntimeError(f"Unknown model {checkpoint}")


# Model weights

def fix_gpt_state_dict(state_dict: StateDict) -> StateDict:
    replacements = {
        'h.': 'blocks.',
        'wte.': 'embed_tokens.',
        'wpe.': 'embed_pos.',
        'attn.c_attn': 'attn.qkv',
        'attn.c_proj': 'attn.out',
        'mlp.c_fc': 'ff.up',
        'mlp.c_proj': 'ff.down',
        'ln_1.': 'ln1.',
        'ln_2.': 'ln2.',
        'ln_f.': 'ln.'}
    linears = ['attn.qkv.weight', 'attn.out.weight', 'ff.up.weight', 'ff.down.weight']
    biases = ['attn.bias', 'attn.masked_bias']

    for src, dst in replacements.items():
        state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
    state_dict = {k:v for k,v in state_dict.items() if not any(x in k for x in biases)}
    state_dict = {k: v.transpose(-1, -2) if any(x in k for x in linears) else v for k,v in state_dict.items()}
    state_dict['out.weight'] = state_dict['embed_tokens.weight']
    for key in list(state_dict.keys()):
        if '.qkv.' in key:
            value = state_dict.pop(key)
            q, k, v = (key.replace('.qkv.', f'.{x}.') for x in 'qkv')
            state_dict[q], state_dict[k], state_dict[v] = value.view(3, value.shape[0] // 3, *value.shape[1:]).unbind(dim=0)
    return state_dict

def fix_llama_state_dict(state_dict: StateDict) -> StateDict:
    replacements = {
        'model.': '',
        'layers.': 'blocks.',
        'input_layernorm': 'ln1',
        'post_attention_layernorm': 'ln2',
        'self_attn.': 'attn.',
        'attn.o_proj.': 'attn.out.',
        'mlp.': 'ff.',
        'norm.': 'ln.',
        'lm_head.': 'out.',
        '_proj.': '.'}
    for src, dst in replacements.items():
        state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
    return state_dict

def load_pretrained_state_dict(checkpoint: str) -> StateDict:
    if checkpoint.startswith('gpt2'):
        checkpoint_path = CHECKPOINT_DIR / f'{checkpoint}.safetensors'
        weights_url = GPT2_URL.format(model=checkpoint)
        state_dict = load_checkpoint(weights_url, checkpoint_path)
        return fix_gpt_state_dict(state_dict)
    elif checkpoint.startswith('llama3'):
        assert HUGGINGFACE_API_KEY, "HUGGINGFACE_API_KEY must be set to download llama models"
        weights_urls = [LLAMA3_URL.format(size=checkpoint.removeprefix('llama3-'), index=i) for i in range(1, 5)]
        checkpoint_paths = [CHECKPOINT_DIR / f'{checkpoint}-{i:05d}.safetensors' for i in range(len(weights_urls))]
        state_dict = load_checkpoints(weights_urls, checkpoint_paths)
        return fix_llama_state_dict(state_dict)
    else:
        raise RuntimeError(f"Unknown model {checkpoint}")


# Load model + config + tokenizer

def load_model_data(checkpoint: str) -> tuple[GPTConfig, Tokenizer, StateDict]:
    if checkpoint in CONFIGS:
        state_dict = load_pretrained_state_dict(checkpoint)
        config = CONFIGS[checkpoint]
        tokenizer = load_pretrained_tokenizer(checkpoint)
        return config, tokenizer, state_dict

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise RuntimeError(f"The checkpoint you specified doesn't exist! Please specify either a valid file path or one of the pretrained checkpoints ({', '.join(CONFIGS)})")
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"No config.yaml found for the specified checkpoint file.")

    schema = OmegaConf.structured(GPTConfig)
    config = OmegaConf.merge(schema, OmegaConf.load(config_path).model)
    config = OmegaConf.to_object(config)

    with safetensors.safe_open(checkpoint, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    tokenizer = load_gpt2_tokenizer()  # TODO: Deal with this properly
    return config, tokenizer, state_dict

def load_model(config: GPTConfig, state_dict: StateDict, device: torch.device) -> GPT:
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)  # TODO: Should probably disable this for float32 models like pretrained GPT2
    model = GPT(config).eval()
    model.load_state_dict(state_dict)
    return model
