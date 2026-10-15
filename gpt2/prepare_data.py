
import os
import argparse

import numpy as np
import tiktoken
from datasets import load_dataset


class ShardWriter:
    """Write token IDs to fixed-size NumPy shards."""

    def __init__(self, output_dir, prefix, shard_size):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_index = 0
        self.buffer = []
        self.total_tokens = 0

    def add_tokens(self, tokens):
        self.total_tokens += len(tokens)
        offset = 0

        while offset < len(tokens):
            space = self.shard_size - len(self.buffer)
            end = min(offset + space, len(tokens))

            self.buffer.extend(tokens[offset:end])
            offset = end

            if len(self.buffer) == self.shard_size:
                self.flush()

    def flush(self):
        if not self.buffer:
            return

        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_{self.shard_index:05d}.npy",
        )

        np.save(
            path,
            np.asarray(self.buffer, dtype=np.uint32),
        )

        print(f"wrote {path}: {len(self.buffer):,} tokens")

        self.shard_index += 1
        self.buffer.clear()

    def close(self):
        # Save the final partial shard as well.
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
    parser.add_argument("--output-dir", default="data_smoke")
    parser.add_argument("--shard-size", type=int, default=1_000_000)
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--val-documents", type=int, default=50)

    args = parser.parse_args()

    if args.shard_size < 2:
        parser.error("--shard-size must be at least 2")

    if args.max_documents < 2:
        parser.error("--max-documents must be at least 2")

    if not 1 <= args.val_documents < args.max_documents:
        parser.error(
            "--val-documents must be at least 1 and "
            "less than --max-documents"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    dataset = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        streaming=True,
    )

    val_writer = ShardWriter(
        args.output_dir, "val", args.shard_size
    )
    train_writer = ShardWriter(
        args.output_dir, "train", args.shard_size
    )

    document_count = 0
    val_documents_written = 0
    train_documents_written = 0

    try:
        for example in dataset:
            text = example.get(args.text_column)

            if not isinstance(text, str) or not text.strip():
                continue

            # GPT-2 tokenizer IDs plus end-of-text marker.
            tokens = enc.encode_ordinary(text)
            tokens.append(enc.eot_token)

            if document_count < args.val_documents:
                val_writer.add_tokens(tokens)
                val_documents_written += 1
            else:
                train_writer.add_tokens(tokens)
                train_documents_written += 1

            document_count += 1

            if document_count >= args.max_documents:
                break

    finally:
        val_writer.close()
        train_writer.close()

    if val_documents_written == 0:
        raise ValueError("No validation documents were produced.")

    if train_documents_written == 0:
        raise ValueError("No training documents were produced.")

    print("\nData preparation complete")
    print(f"Documents processed: {document_count:,}")
    print(f"Training documents: {train_documents_written:,}")
    print(f"Validation documents: {val_documents_written:,}")
    print(f"Training tokens: {train_writer.total_tokens:,}")
    print(f"Validation tokens: {val_writer.total_tokens:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
