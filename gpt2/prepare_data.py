import os
import glob
import json
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


class ShardWriter:
    def __init__(self, output_dir, prefix, shard_size):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_index = 0
        self.buffer = []
        self.buffer_tokens = 0
        self.total_tokens = 0

        os.makedirs(output_dir, exist_ok=True)

    @property
    def token_count(self):
        return self.total_tokens + self.buffer_tokens

    def add_tokens(self, tokens):
        tokens = np.asarray(tokens, dtype=np.uint16)
        offset = 0

        while offset < len(tokens):
            remaining = self.shard_size - self.buffer_tokens
            take = min(remaining, len(tokens) - offset)

            self.buffer.append(
                tokens[offset:offset + take].copy()
            )

            self.buffer_tokens += take
            offset += take

            if self.buffer_tokens == self.shard_size:
                self.flush()

    def flush(self):
        if not self.buffer:
            return

        tokens = np.concatenate(self.buffer)

        filename = os.path.join(
            self.output_dir,
            f"{self.prefix}_{self.shard_index:05d}.npy",
        )

        np.save(filename, tokens)

        self.total_tokens += len(tokens)
        self.shard_index += 1
        self.buffer = []
        self.buffer_tokens = 0

        print(
            f"Saved {filename} "
            f"({len(tokens):,} tokens)"
        )

    def finish(self):
        self.flush()


def prepare_data(
    output_dir,
    max_tokens,
    val_tokens,
    shard_size,
    max_documents,
    overwrite,
):
    if max_tokens <= val_tokens:
        raise ValueError(
            "max_tokens must be greater than val_tokens"
        )

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    if val_tokens < 0:
        raise ValueError("val_tokens cannot be negative")

    os.makedirs(output_dir, exist_ok=True)

    existing_shards = (
        glob.glob(os.path.join(output_dir, "train_*.npy"))
        + glob.glob(os.path.join(output_dir, "val_*.npy"))
    )

    if existing_shards and not overwrite:
        raise FileExistsError(
            f"Existing dataset shards found in {output_dir}. "
            "Use --overwrite to replace them."
        )

    if overwrite:
        for path in existing_shards:
            os.remove(path)

    enc = tiktoken.get_encoding("gpt2")

    train_writer = ShardWriter(
        output_dir,
        "train",
        shard_size,
    )

    val_writer = ShardWriter(
        output_dir,
        "val",
        shard_size,
    )

    train_token_target = max_tokens - val_tokens

    print("Loading FineWeb-Edu...")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Configuration: {DATASET_CONFIG}")
    print(f"Training token target: {train_token_target:,}")
    print(f"Validation token target: {val_tokens:,}")
    print(f"Shard size: {shard_size:,}")

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    documents_processed = 0

    for document in dataset:
        if max_documents and documents_processed >= max_documents:
            break

        text = document.get("text")

        if not text or not text.strip():
            continue

        document_tokens = np.asarray(
            [enc.eot_token] + enc.encode_ordinary(text),
            dtype=np.uint16,
        )

        documents_processed += 1

        if val_writer.token_count < val_tokens:
            val_writer.add_tokens(document_tokens)
        else:
            train_writer.add_tokens(document_tokens)

        if train_writer.token_count >= train_token_target:
            break

        if documents_processed % 1000 == 0:
            print(
                f"Documents: {documents_processed:,} | "
                f"Train: {train_writer.token_count:,} | "
                f"Validation: {val_writer.token_count:,}"
            )

    train_writer.finish()
    val_writer.finish()

    del dataset

    manifest = {
        "dataset": DATASET_NAME,
        "configuration": DATASET_CONFIG,
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "documents_processed": documents_processed,
        "train_tokens": train_writer.total_tokens,
        "val_tokens": val_writer.total_tokens,
        "train_shards": train_writer.shard_index,
        "val_shards": val_writer.shard_index,
        "shard_size": shard_size,
    }

    manifest_path = os.path.join(
        output_dir,
        "metadata.json",
    )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print("\nDataset preparation complete")
    print(json.dumps(manifest, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "data_fineweb",
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10_000_000,
    )

    parser.add_argument(
        "--val-tokens",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    prepare_data(
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        val_tokens=args.val_tokens,
        shard_size=args.shard_size,
        max_documents=args.max_documents,
        overwrite=args.overwrite,
    )
