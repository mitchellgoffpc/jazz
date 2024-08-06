import tiktoken
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, directory_path, context_length, tokenizer=None):
        self.directory_path = Path(directory_path)
        self.context_length = context_length
        self.file_paths = []
        self.file_lengths = []
        self.file_offsets = [0]
        self.tokenizer = tokenizer
        self.token_dtype = np.int32 if self.tokenizer else np.uint8
        self.token_bytes = self.token_dtype().itemsize

        for file_path in self.directory_path.glob('*.txt'):
            tok_file_path = file_path
            if self.tokenizer:
                tok_file_path = file_path.with_suffix('.tok')
                with file_path.open('rb') as f:
                    text = f.read().decode('utf-8')
                    tokens = self.tokenizer.encode(text)
                    with tok_file_path.open('wb') as tf:
                        tf.write(np.array(tokens, dtype=self.token_dtype).tobytes())

            file_length = tok_file_path.stat().st_size // self.token_bytes
            self.file_lengths.append(file_length)
            self.file_offsets.append(self.file_offsets[-1] + file_length)
            self.file_paths.append(tok_file_path)

        self.total_length = self.file_offsets[-1]
        if not self.file_paths:
            raise RuntimeError("No .txt files found in given directory")
        if self.context_length >= self.total_length:
            raise RuntimeError("Dataset size must be greater than context length")

    def __len__(self):
        return self.total_length - self.context_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Find the document containing this index
        doc_idx = next(i for i, offset in enumerate(self.file_offsets) if offset > idx) - 1
        start_offset = self.file_offsets[doc_idx]
        doc_start = idx - start_offset

        token_chunks = []
        remaining_length = self.context_length

        while remaining_length > 0:
            # Read the tokens from this document
            with self.file_paths[doc_idx].open('rb') as f:
                f.seek(doc_start * self.token_bytes)
                chunk = np.frombuffer(f.read(remaining_length * self.token_bytes), dtype=self.token_dtype)
            token_chunks.append(chunk)
            remaining_length -= len(chunk)

            # Move to next document
            doc_idx += 1
            doc_start = 0
            if doc_idx >= len(self.file_paths):
                break

        # Concatenate and pad (padding may not be necessary?)
        tokens = np.concatenate(token_chunks)
        if len(tokens) < self.context_length:
            tokens = np.pad(tokens, (0, self.context_length - len(tokens)), constant_values=0)

        assert len(tokens) == self.context_length
        return tokens.astype(np.int64)


if __name__ == "__main__":
    import sys, random, time, argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Text Dataset Utility")
    parser.add_argument("mode", choices=['show', 'benchmark'], help="Mode of operation")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--context_length", type=int, default=128, help="Context length for the dataset")
    parser.add_argument("--tokenizer", choices=['gpt2'], default=None, help="Tokenizer to use")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer == 'gpt2':
        tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(args.dataset_path, context_length=args.context_length, tokenizer=tokenizer)

    if args.mode == 'show':
        num_samples = 10
        random_indices = random.sample(range(len(dataset)), num_samples)
        for i, idx in enumerate(random_indices):
            sample = dataset[idx]
            text = tokenizer.decode(sample.tolist()) if tokenizer else sample.tobytes().decode('utf-8', errors='replace')
            print(f"Sample {i + 1}:")
            print(text)
            print("--------")
    
    elif args.mode == 'benchmark':
        num_samples = 1000
        start_time = time.time()
        
        for _ in tqdm(range(num_samples), desc="Benchmarking"):
            idx = random.randint(0, len(dataset) - 1)
            _ = dataset[idx]
        
        end_time = time.time()
        total_time = end_time - start_time
        samples_per_second = num_samples / total_time
        
        print(f"Benchmark results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Samples per second: {samples_per_second:.2f}")
