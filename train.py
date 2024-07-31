#!/usr/bin/env python
import os
import csv
import time
import shutil
import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Any
from omegaconf import OmegaConf
from dataclasses import dataclass, field, replace
from safetensors.torch import save_file

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models.gpt import GPT, GPTConfig
from models.llama import Llama, LlamaConfig
from datasets.text import TextDataset

OPTIMIZERS = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}
MODEL_TYPES = {GPTConfig: GPT, LlamaConfig: Llama}
CONFIGS = {
    'gpt2': GPTConfig(num_layers=12, num_heads=12, embed_size=768),
    'gpt2-medium': GPTConfig(num_layers=24, num_heads=16, embed_size=1024),
    'gpt2-large': GPTConfig(num_layers=36, num_heads=20, embed_size=1280),
    'gpt2-xl': GPTConfig(num_layers=48, num_heads=25, embed_size=1600),
    'llama-160m': LlamaConfig(num_layers=12, num_heads=12, num_kv_heads=12, embed_size=768, intermediate_size=2048, context_size=1024),
    'llama-400m': LlamaConfig(num_layers=24, num_heads=16, num_kv_heads=16, embed_size=1024, intermediate_size=2816, context_size=1024),
    'llama3-8b': LlamaConfig(num_layers=32, num_heads=32, embed_size=4096)}

@dataclass
class Config:
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    val_split: float = 0.1
    optim: str = 'Adam'
    type: str = 'gpt2'
    model: Any = field(default_factory=dict)

    data_path: str = '/raid.unprotected/datasets/shakespeare'
    checkpoint_path: Optional[str] = None
    save: bool = True
    save_every: int = 1


def get_dataloader(config, rank, train=True):
    dataset = TextDataset(config.data_path, config.model.context_size + 1)
    num_samples = len(dataset) // (config.model.context_size + 1)
    indices = torch.randperm(len(dataset))[:num_samples]

    split_idx = int(len(indices) * (1 - config.val_split))
    subset_indices = indices[:split_idx] if train else indices[split_idx:]
    subset = Subset(dataset, subset_indices)
    sampler = DistributedSampler(subset, rank=rank)
    return DataLoader(subset, batch_size=config.batch_size, num_workers=4, pin_memory=True, sampler=sampler)

def all_reduce(data, device):
    data = torch.tensor(data, device=device)
    dist.all_reduce(data)
    return data.item()

def train(rank, world_size, config, result_path):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Instantiate the model
    device = torch.device(f'cuda:{rank}')
    config.model = replace(CONFIGS[config.type], **config.model)
    model = MODEL_TYPES[type(config.model)](config.model).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = OPTIMIZERS[config.optim](model.parameters(), lr=config.learning_rate)
    if config.checkpoint_path:
        model.load_state_dict(torch.load(config.checkpoint_path))
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

    # Load the dataset
    train_loader = get_dataloader(config, rank, train=True)
    val_loader = get_dataloader(config, rank, train=False)

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
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'epoch_duration'])

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()

        for tokens in (pbar := tqdm(train_loader, leave=False, disable=rank>0)):
            start_time = time.time()
            tokens = tokens.to(device)

            optimizer.zero_grad()
            outputs = model(tokens[:, :-1])
            loss = F.cross_entropy(outputs.flatten(end_dim=1), tokens[:, 1:].flatten())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            step_time = (time.time() - start_time) * 1000
            pbar.set_description(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Step Time: {step_time:.2f}ms")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tokens in (pbar := tqdm(val_loader, leave=False, disable=rank>0)):
                tokens = tokens.to(device)
                outputs = model(tokens[:, :-1])
                loss = F.cross_entropy(outputs.flatten(end_dim=1), tokens[:, 1:].flatten())
                val_loss += loss.item()
                pbar.set_description(f"Epoch {epoch} | Val Loss: {loss.item():.4f}")

        # Print report and write results to CSV file
        train_loss, val_loss = (all_reduce(x, device) for x in (train_loss, val_loss))

        train_loss = train_loss / (len(train_loader) * world_size)
        val_loss = val_loss / (len(val_loader) * world_size)
        epoch_duration = int(time.time() - epoch_start_time)

        if rank == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Duration: {datetime.timedelta(seconds=epoch_duration)}")

        if save_experiment:
            with open(result_path / 'results.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, epoch_duration])

        # Save the model checkpoint
        if save_experiment and epoch % config.save_every == 0:
            state_dict = {k.removeprefix('module.'): v.cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, result_path / f'checkpoint_{epoch}.safetensors')


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
