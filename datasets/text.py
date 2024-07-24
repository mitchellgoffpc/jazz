import tiktoken
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, directory_path, context_length):
        self.directory_path = Path(directory_path)
        self.context_length = context_length
        self.file_paths = []
        self.file_lengths = []
        self.file_offsets = [0]
        self.tokenizer = tiktoken.get_encoding("gpt2")

        for file_path in self.directory_path.glob('*.txt'):
            self.file_paths.append(file_path)
            tok_file_path = file_path.with_suffix('.tok')

            # Tokenize
            if not tok_file_path.exists():
                with file_path.open('r', encoding='utf-8') as f:
                    text = f.read()
                tokens = self.tokenizer.encode(text)
                with tok_file_path.open('wb') as f:
                    f.write(np.array(tokens, dtype=np.int32).tobytes())

            # Compute lengths + offsets
            file_length = tok_file_path.stat().st_size // 4  # Each token is 4 bytes (int32)
            self.file_lengths.append(file_length)
            self.file_offsets.append(self.file_offsets[-1] + file_length)

        self.total_length = self.file_offsets[-1]
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
            tok_file_path = self.file_paths[doc_idx].with_suffix('.tok')
            with tok_file_path.open('rb') as f:
                f.seek(doc_start * 4)  # Each token is 4 bytes
                chunk = np.frombuffer(f.read(remaining_length * 4), dtype=np.int32)
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
    args = parser.parse_args()

    dataset = TextDataset(args.dataset_path, context_length=args.context_length)

    if args.mode == 'show':
        num_samples = 10
        random_indices = random.sample(range(len(dataset)), num_samples)
        for i, idx in enumerate(random_indices):
            sample = dataset[idx]
            text = dataset.tokenizer.decode(sample.tolist())
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