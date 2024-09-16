
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from helpers import download_file, load_model_data, load_model

DATA_CACHE_DIR = Path(__file__).parent / "hellaswag"
DATA_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def load_examples(split):
    download_file(DATA_URLS[split], DATA_CACHE_DIR / f"hellaswag_{split}.jsonl")
    with open(DATA_CACHE_DIR / f"hellaswag_{split}.jsonl", "r") as f:
        for line in f:
            yield json.loads(line)

def render_example(tokenizer, example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """

    # gather up all the tokens
    ctx_tokens = tokenizer.encode(example["ctx"])
    tok_rows = []
    mask_rows = []
    for end in example["endings"]:
        end_tokens = tokenizer.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, example["label"]

@torch.no_grad
def evaluate_example(model, device, tokens, mask):
    tokens = tokens.to(device)
    mask = mask.to(device)
    B, _ = tokens.shape

    logits = model(tokens[:, :-1])  # evaluate the loss at all positions
    losses = F.cross_entropy(logits.flatten(end_dim=1), tokens[:, 1:].flatten(), reduction='none').view(B, -1)
    completion_losses = losses * mask[:, 1:]  # mask out the losses for the prompt tokens
    return completion_losses.sum(dim=1) / mask[:, 1:].sum(dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model checkpoint name or path")
    args = parser.parse_args()

    device = torch.device('cuda')
    config, tokenizer, state_dict = load_model_data(args.model)
    model = load_model(config, state_dict, device)

    num_correct = num_total = 0
    for example in load_examples("val"):
        tokens, mask, label = render_example(tokenizer, example)
        losses = evaluate_example(model, device, tokens, mask)
        pred = losses.argmin().item()
        num_correct += int(pred == label)
        num_total += 1

        if num_total % 100 == 0:
            print(f"{num_total} acc_norm: {num_correct}/{num_total}={num_correct/num_total:.4f}")
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {losses[i].item():.4f}) {end}")
            print(f"predicted: {pred}, actual: {label}")