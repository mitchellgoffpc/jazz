#!/usr/bin/env python
import os
import csv
import math
import time
import shutil
import datetime
import itertools
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from safetensors.torch import save_file

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models.gpt import GPT, GPTConfig
from datasets.text import TextDataset
from data.hellaswag import load_examples, render_example, evaluate_example
from helpers import load_pretrained_tokenizer

DTYPES = {'f32': torch.float32, 'bf16': torch.bfloat16}

@dataclass
class Config:
    num_steps: int = 10000
    num_val_steps: int = 10
    batch_size: int = 64
    grad_accum_steps: int = 8
    learning_rate: float = 6e-4
    learning_rate_decay: bool = True
    weight_decay: float = 0.1
    num_workers: int = 4
    model: GPTConfig = field(default_factory=GPTConfig)
    data_path: str = str(Path(__file__).parent / 'data' / 'edu_fineweb10B')
    tokenizer: Optional[str] = 'gpt2'
    checkpoint_path: Optional[str] = None
    save: bool = True
    save_every: int = 1000  # steps
    eval_every: int = 250  # steps
    dtype: str = 'f32'
    compile: bool = False


def get_dataloader(config, rank, split):
    while True:
        dataset = TextDataset(config.data_path, config.model.context_size, config.tokenizer, split=split)  # recreate the dataset for different offsets
        sampler = DistributedSampler(dataset, rank=rank)
        yield from DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, sampler=sampler)

def all_reduce(data, device):
    data = torch.tensor(data, device=device)
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    return data.item()

def train(rank, world_size, config, result_path):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Load the dataset
    config.model.vocab_size = {'llama3': 128256, 'gpt2': 50304, None: 256}[config.tokenizer]
    train_loader = get_dataloader(config, rank, split='train')
    val_loader = get_dataloader(config, rank, split='val')
    tokenizer = load_pretrained_tokenizer(config.tokenizer)

    # Instantiate the model and optimizer
    torch.set_float32_matmul_precision('high')
    device = torch.device(f'cuda:{rank}')
    raw_model = GPT(config.model).to(device)
    model = torch.compile(raw_model) if config.compile else raw_model
    model = DDP(model, device_ids=[rank])
    if config.checkpoint_path:
        model.load_state_dict(torch.load(config.checkpoint_path))

    optim_groups = [
        {'params': [p for p in model.parameters() if p.dim() >= 2], 'weight_decay': config.weight_decay},
        {'params': [p for p in model.parameters() if p.dim() < 2], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)

    assert config.batch_size % config.grad_accum_steps == 0, "batch_size must be a multiple of grad_accum_steps"
    B = config.batch_size // config.grad_accum_steps
    dtype = DTYPES[config.dtype]
    use_amp = dtype is not torch.float32
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

    # Create results directory and csv file
    save_experiment = config.save and rank == 0
    if save_experiment:
        code_path = result_path / 'code'
        code_path.mkdir(parents=True, exist_ok=True)

        for py_file in Path(__file__).parent.glob('*.py'):
            shutil.copy(py_file, code_path)
        with open(result_path / 'config.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(config))
        with open(result_path / 'results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'key', 'value'])

    # Helper functions
    def log_values(step, values):
        with open(result_path / 'results.csv', 'a') as f:
            writer = csv.writer(f)
            for key, value in values.items():
                writer.writerow([step, key, value])

    def get_lr(step):  # credit to karpathy's build-nanogpt
        max_lr, min_lr = config.learning_rate, config.learning_rate * 0.1
        warmup_steps = 375*1000*1000 // (world_size * config.batch_size * config.model.context_size)  # 375M tokens
        decay_steps = config.num_steps  # 10*1000*1000*1000 // (world_size * config.batch_size * config.model.context_size)  # 10B tokens
        if not config.learning_rate_decay:  # no decay
            return max_lr
        elif step < warmup_steps:  # warmup
            return max_lr * (step+1) / warmup_steps
        elif step > decay_steps:  # clip at min_lr
            return min_lr
        else:  # cosine decay
            decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
            return min_lr + coeff * (max_lr - min_lr)

    def train_step(tokens, step):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for i in range(config.grad_accum_steps):
            batch_tokens = tokens[i*B : i*B+B]
            model.require_backward_grad_sync = (i == config.grad_accum_steps - 1)
            with torch.amp.autocast(enabled=use_amp, dtype=dtype, device_type=device.type):
                outputs = model(batch_tokens[:, :-1])
                loss = F.cross_entropy(outputs.flatten(end_dim=1), batch_tokens[:, 1:].flatten()) / config.grad_accum_steps
            loss.backward()
            total_loss += loss.item()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        total_loss = all_reduce(total_loss, device) / world_size
        norm = all_reduce(norm, device) / world_size
        return total_loss, norm, lr

    @torch.no_grad
    def val_step(tokens):
        model.eval()
        total_loss = 0.0
        for i in range(config.grad_accum_steps):
            batch_tokens = tokens[i*B : i*B+B]
            with torch.amp.autocast(enabled=use_amp, dtype=dtype, device_type=device.type):
                outputs = model(batch_tokens[:, :-1])
                total_loss += F.cross_entropy(outputs.flatten(end_dim=1), batch_tokens[:, 1:].flatten()).item() / config.grad_accum_steps
        return all_reduce(total_loss, device) / world_size

    @torch.no_grad
    def eval_hellaswag():
        num_correct = num_total = 0
        for example in itertools.islice(load_examples('val'), rank, None, world_size):
            tokens, mask, label = render_example(tokenizer, example)
            losses = evaluate_example(raw_model, device, tokens, mask)  # don't use the compiled model for hellaswag, it busts the cache
            num_correct += int(losses.argmin().item() == label)
            num_total += 1
        return all_reduce(num_correct, device) / all_reduce(num_total, device)

    # Training loop
    for step in range(config.num_steps):
        # Run a single train step
        start_time = time.perf_counter()
        tokens = next(train_loader).to(device)
        loss, norm, lr = train_step(tokens, step=step)
        step_time = time.perf_counter() - start_time
        tokens_per_sec = (world_size * config.batch_size * config.model.context_size) / step_time

        if rank == 0:
            print(f"step: {step:6d} | loss: {loss:.6f} | norm: {norm:.4f} | lr: {lr:.4e} | dt: {step_time*1000:.1f}ms | tok/s: {tokens_per_sec:.1f}")
        if save_experiment:
            log_values(step, {'train_loss': loss, 'norm': norm, 'lr': lr, 'step_time': step_time})

        # Periodically compute evals
        if step and (step % config.eval_every == 0 or step == config.num_steps - 1):
            if rank == 0:
                print('---\n... running evals ...', end='\r')

            start_time = time.perf_counter()
            # val_loss = sum(val_step(tokens.to(device)) for tokens in itertools.islice(val_loader, config.num_val_steps)) / config.num_val_steps
            val_loss = 0.0
            hellaswag_accuracy = eval_hellaswag()
            eval_time = time.perf_counter() - start_time

            if rank == 0:
                print(f"step: {step:6d} | val loss: {val_loss:.6f} | hellaswag: {hellaswag_accuracy:.4f} | eval time: {eval_time:.1f}s\n---")
            if save_experiment:
                log_values(step, {'val_loss': val_loss, 'hellaswag': hellaswag_accuracy})

        # Periodically save the model
        if save_experiment and step and (step % config.save_every == 0 or step == config.num_steps - 1):
            state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
            save_file(state_dict, result_path / f'checkpoint_{step:06d}.safetensors')


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, OmegaConf.from_cli())
    config = OmegaConf.to_object(config)
    ngpus = torch.cuda.device_count()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = Path(__file__).parent / 'experiments' / current_time

    if ngpus > 1:
        mp.spawn(train, args=(ngpus, config, result_path), nprocs=ngpus, join=True)
    else:
        train(0, 1, config, result_path)
