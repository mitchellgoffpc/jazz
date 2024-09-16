#!/usr/bin/env python
import time
import torch
import argparse
from tqdm import trange
from helpers import load_model_data, load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT model generator')
    parser.add_argument('model', help='Model checkpoint name or path')
    parser.add_argument('-b', '--benchmark', action='store_true', help='Run a benchmark')
    parser.add_argument('-p', '--prompt', help='Prompt to generate completion for')
    args = parser.parse_args()

    st = time.time()
    device = torch.device('cuda')
    config, tokenizer, state_dict = load_model_data(args.model)
    model = load_model(config, state_dict, device)
    print(f"Loaded model in {time.time() - st:.2f}s")

    # Benchmark
    if args.benchmark:
        large_context = torch.randint(0, 50257, size=(1, 1024)).to(device)
        with torch.no_grad():
            for i in trange(1024, desc="benchmarking bs=1, seqlen=1"):
                data = model(large_context[:,i:i+1])

    # Decode
    else:
        prompt = args.prompt or "The capital of Germany is Berlin. The capital of France is"
        tokens = tokenizer.encode(prompt)
        context = torch.tensor(tokens)[None].to(device)
        result = model.generate(context, num_tokens=3, top_k=10)
        print(f"Prompt:    ", prompt)
        print(f"Completion:", tokenizer.decode(result[0].tolist()))
