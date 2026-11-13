
import os
import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F


DATASET_NAME = "allenai/hellaswag"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

SPLIT_NAMES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}

enc = tiktoken.get_encoding("gpt2")


def download(split):
    if split not in SPLIT_NAMES:
        raise ValueError(
            f"Invalid split: {split}. Choose from {list(SPLIT_NAMES)}"
        )

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAMES[split],
        cache_dir=DATA_CACHE_DIR,
    )

    return dataset


def render_example(example):
    ctx = example["ctx"]
    endings = example["endings"]
    label = int(example["label"])

    if len(endings) != 4:
        raise ValueError(f"Expected 4 endings, received {len(endings)}")

    if not 0 <= label < 4:
        raise ValueError(f"Invalid label: {label}")

    ctx_tokens = enc.encode(ctx)

    data = {
        "label": label,
        "ctx_tokens": ctx_tokens,
        "ending_tokens": [],
    }

    token_rows = []
    mask_rows = []

    for ending in endings:
        ending_tokens = enc.encode(" " + ending)

        data["ending_tokens"].append(ending_tokens)

        token_rows.append(ctx_tokens + ending_tokens)
        mask_rows.append(
            [0] * len(ctx_tokens) + [1] * len(ending_tokens)
        )

    max_len = max(len(row) for row in token_rows)

    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (token_row, mask_row) in enumerate(
        zip(token_rows, mask_rows)
    ):
        tokens[i, :len(token_row)] = torch.tensor(token_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    dataset = download(split)

    for example in tqdm(dataset, desc=f"HellaSwag {split}"):
        yield example


@torch.no_grad()
def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[:, :-1, :]
    shift_tokens = tokens[:, 1:]
    shift_mask = mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_tokens.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * shift_mask

    average_loss = -token_log_probs.sum(dim=1) / shift_mask.sum(dim=1)

    return average_loss.argmin().item(), average_loss


@torch.no_grad()
def evaluate(
    model,
    split="val",
    device="cuda",
    max_examples=None,
    distributed=False,
    rank=0,
    world_size=1,
):
    model.eval()

    total = 0
    correct = 0

    for example in iterate_examples(split):
        data, tokens, mask, label = render_example(example)

        tokens = tokens.to(device)
        mask = mask.to(device)

        logits, _ = model(tokens)

        prediction, scores = get_most_likely_row(
            tokens,
            mask,
            logits,
        )

        total += 1
        correct += int(prediction == label)

        if max_examples is not None and total >= max_examples:
            break

    accuracy = correct / total

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
