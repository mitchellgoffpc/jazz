import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, directory_path, context_length, tokenizer=None, split='train'):
        self.directory_path = Path(directory_path)
        self.context_length = context_length
        self.offset = np.random.randint(0, context_length)
        self.file_paths = []
        self.file_lengths = []
        self.file_offsets = [0]
        self.file_ext = 'bin' if tokenizer else 'txt'
        self.token_dtype = {'llama3': np.uint32, 'gpt2': np.uint16, None: np.uint8}[tokenizer]
        self.token_bytes = self.token_dtype().itemsize
        self.header_size = 256 * 4 if tokenizer else 0
        self.header_magic = {'llama3': 20240801, 'gpt2': 20240520, None: None}[tokenizer]

        self.file_paths = sorted(self.directory_path.glob(f"*_{split}_*.{self.file_ext}"))
        for tok_file_path in self.file_paths:
            file_length = (tok_file_path.stat().st_size - self.header_size) // self.token_bytes
            if tokenizer:
                with open(tok_file_path, 'rb') as f:
                    header = np.frombuffer(f.read(self.header_size), dtype=np.int32)
                assert header[0] == self.header_magic, f'Invalid header in file {tok_file_path}'
                assert header[2] == file_length, f'Mismatch between file size and header size in file {tok_file_path}'
            self.file_lengths.append(file_length)
            self.file_offsets.append(self.file_offsets[-1] + file_length)

        self.total_length = self.file_offsets[-1]
        if not self.file_paths:
            raise RuntimeError(f"No .{self.file_ext} files found in given directory")
        if self.context_length >= self.total_length:
            raise RuntimeError("Dataset size must be greater than context length")

    def __len__(self):
        return (self.total_length - self.offset - 1) // self.context_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Find the document containing this index
        token_idx = (idx * self.context_length) + self.offset
        doc_idx = next(i for i, offset in enumerate(self.file_offsets) if offset > token_idx) - 1
        start_offset = self.file_offsets[doc_idx]
        doc_start = token_idx - start_offset

        token_chunks = []
        remaining_length = self.context_length + 1

        while remaining_length > 0:
            # Read the tokens from this document
            with self.file_paths[doc_idx].open('rb') as f:
                f.seek(self.header_size + doc_start * self.token_bytes)
                chunk = np.frombuffer(f.read(remaining_length * self.token_bytes), dtype=self.token_dtype)
            token_chunks.append(chunk)
            remaining_length -= len(chunk)

            # Move to next document
            doc_idx += 1
            doc_start = 0
            if doc_idx >= len(self.file_paths) and remaining_length > 0:
                raise IndexError("Attempted to read past the end of the last file, this should never happen!!!")

        # Concatenate and pad (padding may not be necessary?)
        tokens = np.concatenate(token_chunks)
        if len(tokens) < self.context_length + 1:
            tokens = np.pad(tokens, (0, self.context_length - len(tokens)), constant_values=0)

        assert len(tokens) == self.context_length + 1
        return tokens.astype(np.int64)


if __name__ == "__main__":
    import sys, random, time, argparse, tiktoken
    from tqdm import trange

    parser = argparse.ArgumentParser(description="Text Dataset Utility")
    parser.add_argument("mode", choices=['show', 'benchmark'], help="Mode of operation")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--context_length", type=int, default=128, help="Context length for the dataset")
    parser.add_argument("--tokenizer", choices=['gpt2', 'llama3'], default=None, help="Tokenizer to use")
    args = parser.parse_args()

    dataset = TextDataset(args.dataset_path, context_length=args.context_length, tokenizer=args.tokenizer)

    if args.mode == 'show':
        tokenizer = None
        if args.tokenizer == 'gpt2':
            tokenizer = tiktoken.get_encoding("gpt2")
        elif args.tokenizer == 'llama3':
            raise ValueError("De-tokenization for llama3 isn't supported yet")

        print(f"Dataset size: {len(dataset) * args.context_length:,} tokens")
        num_samples = 10
        random_indices = random.sample(range(len(dataset)), num_samples)
        for i, idx in enumerate(random_indices):
            sample = dataset[idx]
            text = tokenizer.decode(sample.tolist()) if tokenizer else sample.tobytes().decode('utf-8', errors='replace')
            print(f"Sample {i + 1}:")
            print(text)
            print("--------")

    elif args.mode == 'benchmark':
        num_samples = 10000
        start_time = time.time()

        for _ in trange(num_samples, desc="Benchmarking"):
            idx = random.randint(0, len(dataset) - 1)
            _ = dataset[idx]

        end_time = time.time()
        total_time = end_time - start_time
        samples_per_second = num_samples / total_time

        print(f"Benchmark results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Samples per second: {samples_per_second:,.2f}")
        print(f"Tokens per second: {samples_per_second*args.context_length:,.2f}")
