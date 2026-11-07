import os
import argparse

import numpy as np
import tiktoken
from datasets import load_dataset


class ShardWriter:
    def __init__(self, output_dir, prefix, shard_size):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_index = 0
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.position = 0
        self.total_tokens = 0

    def add_tokens(self, tokens):
        tokens = np.asarray(tokens, dtype=np.uint16)
        self.total_tokens += len(tokens)

        offset = 0

        while offset < len(tokens):
            space = self.shard_size - self.position
            count = min(space, len(tokens) - offset)

            self.buffer[self.position:self.position + count] = (
                tokens[offset:offset + count]
            )

            self.position += count
            offset += count

            if self.position == self.shard_size:
                self.flush()

    def flush(self):
        if self.position == 0:
            return

        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_{self.shard_index:05d}.npy",
        )

        np.save(path, self.buffer[:self.position])

        print(f"Wrote {path}: {self.position:,} tokens")

        self.shard_index += 1
        self.position = 0

    def close(self):
        self.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
    )
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="data_fineweb")
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--val-tokens", type=int, default=100_000_000)

    args = parser.parse_args()

    if args.shard_size < 2:
        parser.error("--shard-size must be at least 2")

    if args.max_documents < 0:
        parser.error("--max-documents cannot be negative")

    if args.val_tokens < 1:
        parser.error("--val-tokens must be positive")

    os.makedirs(args.output_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    dataset = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        streaming=True,
    )

    val_writer = ShardWriter(
        args.output_dir,
        "val",
        args.shard_size,
    )

    train_writer = ShardWriter(
        args.output_dir,
        "train",
        args.shard_size,
    )

    document_count = 0

    try:
        for example in dataset:
            text = example.get(args.text_column)

            if not isinstance(text, str) or not text.strip():
                continue

            tokens = [enc.eot_token] + enc.encode_ordinary(text)

            remaining_val_tokens = max(
                0,
                args.val_tokens - val_writer.total_tokens,
            )

            val_count = min(len(tokens), remaining_val_tokens)

            if val_count:
                val_writer.add_tokens(tokens[:val_count])

            if val_count < len(tokens):
                train_writer.add_tokens(tokens[val_count:])

            document_count += 1

            if args.max_documents and document_count >= args.max_documents:
                break

    finally:
        val_writer.close()
        train_writer.close()

    if val_writer.total_tokens == 0:
        raise ValueError("No validation tokens were produced.")

    if train_writer.total_tokens == 0:
        raise ValueError("No training tokens were produced.")

    print("\nData preparation complete")
    print(f"Documents processed: {document_count:,}")
    print(f"Training tokens: {train_writer.total_tokens:,}")
    print(f"Validation tokens: {val_writer.total_tokens:,}")
    print(f"Training shards: {train_writer.shard_index:,}")
    print(f"Validation shards: {val_writer.shard_index:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
