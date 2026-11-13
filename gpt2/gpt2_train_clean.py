import os
import math
import time
import glob
import inspect
from bisect import bisect_right
from contextlib import nullcontext
import tiktoken

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hellaswag
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SAVE_INTERVAL = 1000
RESUME_FROM = None

# config

SEED = 1337

DATA_DIR = "data_smoke"
VOCAB_SIZE = 50304
ORIGINAL_VOCAB_SIZE = 50257

BLOCK_SIZE = 16
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4

MAX_STEPS = 3
HELLASWAG_EVAL_INTERVAL = 2
HELLASWAG_EVAL_EXAMPLES = 10
EVAL_INTERVAL = 10
EVAL_ITERS = 3

MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 2
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0

DROPOUT = 0.0
USE_COMPILE = True

enc = tiktoken.get_encoding("gpt2")

SAMPLE_INTERVAL = 1000
SAMPLE_MAX_NEW_TOKENS = 80
SAMPLE_TOP_K = 50

SAMPLE_PROMPTS = [
    "Once upon a time",
    "The future of artificial intelligence",
    "In a distant land",
]

def setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP in this script requires CUDA and NCCL.")

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# Model

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if use_flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0,
                float("-inf"),
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.c_proj(y)
        return self.resid_dropout(y)


class GPT2MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class GPT2Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = GPT2MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
    ):
        super().__init__()

        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "wpe": nn.Embedding(block_size, n_embd),
            "h": nn.ModuleList([
                GPT2Block(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]),
            "ln_f": LayerNorm(n_embd),
        })

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.transformer["wte"].weight

        self.apply(self._init_weights)

        for block in self.transformer["h"]:
            nn.init.normal_(
                block.attn.c_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layer),
            )
            nn.init.normal_(
                block.mlp.c_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layer),
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block size {self.block_size}"
            )

        positions = torch.arange(seq_len, device=idx.device)
        x = self.transformer["wte"](idx) + self.transformer["wpe"](positions)

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50):
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :ORIGINAL_VOCAB_SIZE]

            k = min(top_k, logits.size(-1))
            topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(topk_indices, -1, sampled)
            idx = torch.cat((idx, next_token), dim=1)

        self.train(was_training)
        return idx


# Data loading

class DataLoaderLite:
    def __init__(
        self,
        B,
        T,
        data_dir,
        split="train",
        process_rank=0,
        num_processes=1,
        device="cpu",
    ):
        if B < 1 or T < 1:
            raise ValueError("Batch size and sequence length must be positive.")
        if not 0 <= process_rank < num_processes:
            raise ValueError("process_rank must be within num_processes")

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = torch.device(device)

        self.shard_paths = sorted(
            glob.glob(os.path.join(data_dir, f"{split}_*.npy"))
        )

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No {split} shards found in '{data_dir}'. "
                f"Expected files like {split}_00000.npy."
            )

        self.shards = [
            np.load(path, mmap_mode="r")
            for path in self.shard_paths
        ]

        self.shard_lengths = [len(shard) for shard in self.shards]
        if any(length == 0 for length in self.shard_lengths):
            raise ValueError("One or more data shards are empty.")

        self.cumulative_lengths = np.cumsum(self.shard_lengths).tolist()
        self.total_tokens = self.cumulative_lengths[-1]

        if self.total_tokens < self.num_processes * self.B * self.T + 1:
            raise ValueError("Not enough tokens for all distributed ranks.")

        self.reset()

    def reset(self):
        self.position = self.process_rank * self.B * self.T

    def _read_tokens(self, start, count):
        pieces = []
        position = start % self.total_tokens
        remaining = count

        while remaining > 0:
            shard_index = bisect_right(self.cumulative_lengths, position)
            shard_start = (
                0 if shard_index == 0
                else self.cumulative_lengths[shard_index - 1]
            )

            offset = position - shard_start
            shard = self.shards[shard_index]
            take = min(remaining, len(shard) - offset)

            piece = np.array(
                shard[offset:offset + take],
                dtype=np.int64,
                copy=True,
            )
            pieces.append(torch.from_numpy(piece))

            remaining -= take
            position = (position + take) % self.total_tokens

        return torch.cat(pieces)

    def next_batch(self):
        tokens_per_batch = self.B * self.T

        epoch_start = self.position - self.process_rank * tokens_per_batch

        last_rank_position = (
            epoch_start + (self.num_processes - 1) * tokens_per_batch
        )

        if last_rank_position + tokens_per_batch + 1 > self.total_tokens:
            self.reset()

        buf = self._read_tokens(
            self.position,
            tokens_per_batch + 1,
        )

        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.position += tokens_per_batch * self.num_processes

        return x.to(self.device), y.to(self.device)


# Optimizer and learning rate

def configure_optimizer(model, learning_rate, weight_decay, device):
    parameters = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    decay_params = [
        param for param in parameters.values()
        if param.dim() >= 2
    ]
    no_decay_params = [
        param for param in parameters.values()
        if param.dim() < 2
    ]

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    fused_available = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    use_fused = fused_available and device.type == "cuda"

    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    if fused_available:
        optimizer_kwargs["fused"] = use_fused

    return torch.optim.AdamW(groups, **optimizer_kwargs)


def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    if step >= MAX_STEPS:
        return MIN_LR

    decay_steps = max(1, MAX_STEPS - WARMUP_STEPS)
    decay_ratio = (step - WARMUP_STEPS) / decay_steps
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# Evaluation

@torch.no_grad()
def estimate_loss(model, loader, eval_iters, distributed, device, is_master):
    was_training = model.training
    model.eval()

    try:
        total_loss = torch.zeros((), device=device)

        for _ in range(eval_iters):
            x, y = loader.next_batch()
            _, loss = model(x, y)
            total_loss += loss.detach()

        mean_loss = total_loss / eval_iters

        if distributed:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= dist.get_world_size()

        return mean_loss.item()

    finally:
        model.train(was_training)

@torch.no_grad()
def generate_samples(model, device):
    was_training = model.training
    model.eval()

    try:
        for prompt in SAMPLE_PROMPTS:
            tokens = enc.encode(prompt)

            idx = torch.tensor(
                tokens,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            output = model.generate(
                idx,
                max_new_tokens=SAMPLE_MAX_NEW_TOKENS,
                top_k=SAMPLE_TOP_K,
            )

            generated_text = enc.decode(output[0].tolist())

            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")

    finally:
        model.train(was_training)

def get_raw_model(model):
    if hasattr(model, "module"):
        model = model.module

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model


def save_checkpoint(model, optimizer, scaler, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        "step": step,
        "model": get_raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"checkpoint_{step:08d}.pt",
    )

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    get_raw_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint["step"]

# Training



def train():

    distributed, rank, local_rank, world_size, device = setup_distributed()
    is_master = rank == 0

    torch.manual_seed(SEED)

    global use_flash
    use_flash = device.type == "cuda"

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    if is_master:
        print(f"Device: {device}")
        print(f"Distributed: {distributed}")
        print(f"World size: {world_size}")
        print(f"AMP: {use_amp} ({amp_dtype if use_amp else 'disabled'})")
        print(f"SDPA path: {use_flash}")

    model = GPT2(
        vocab_size=VOCAB_SIZE,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=DROPOUT,
    ).to(device)

    if USE_COMPILE and device.type == "cuda" and not distributed:
        model = torch.compile(model)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    torch.manual_seed(SEED + rank)

    optimizer = configure_optimizer(
        model,
        learning_rate=MAX_LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    start_step = 0

    if RESUME_FROM is not None:
        start_step = load_checkpoint(
            model,
            optimizer,
            scaler,
            RESUME_FROM,
            device,
        )

        if is_master:
            print(f"Resuming training from step {start_step}")

    train_loader = DataLoaderLite(
        B=MICRO_BATCH_SIZE,
        T=BLOCK_SIZE,
        data_dir=DATA_DIR,
        split="train",
        process_rank=rank,
        num_processes=world_size,
        device=device,
    )

    val_loader = DataLoaderLite(
        B=MICRO_BATCH_SIZE,
        T=BLOCK_SIZE,
        data_dir=DATA_DIR,
        split="val",
        process_rank=rank,
        num_processes=world_size,
        device=device,
    )

    model.train()

    for step in range(start_step, MAX_STEPS):

        learning_rate = get_lr(step)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)

        step_loss = torch.zeros((), device=device)
        start_time = time.perf_counter()

        for micro_step in range(GRAD_ACCUM_STEPS):

            x, y = train_loader.next_batch()

            sync_context = (
                model.no_sync()
                if distributed and micro_step < GRAD_ACCUM_STEPS - 1
                else nullcontext()
            )

            with sync_context:

                if use_amp:
                    autocast_context = torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                    )
                else:
                    autocast_context = nullcontext()

                with autocast_context:
                    _, loss = model(x, y)

                step_loss += loss.detach()
                loss = loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        mean_step_loss = step_loss / GRAD_ACCUM_STEPS

        if distributed:
            dist.all_reduce(mean_step_loss, op=dist.ReduceOp.SUM)
            mean_step_loss /= world_size

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if is_master:
            print(
                f"step {step:05d} | "
                f"loss {mean_step_loss.item():.4f} | "
                f"lr {learning_rate:.2e} | "
                f"grad_norm {grad_norm.item():.4f} | "
                f"{elapsed_ms:.2f} ms"
            )

        # Validation loss evaluation
        if (step + 1) % EVAL_INTERVAL == 0:

            validation_loss = estimate_loss(
                model,
                val_loader,
                EVAL_ITERS,
                distributed,
                device,
                is_master,
            )

            if is_master:
                print(f"validation loss: {validation_loss:.4f}")

        # Periodic HellaSwag evaluation
        if (step + 1) % HELLASWAG_EVAL_INTERVAL == 0:

            results = hellaswag.evaluate(
                model,
                split="val",
                device=device,
                max_examples=HELLASWAG_EVAL_EXAMPLES,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )

            if is_master:
                print(
                    f"HellaSwag accuracy: "
                    f"{results['accuracy'] * 100:.2f}% "
                    f"({results['correct']}/{results['total']})"
                )

        if is_master and (step + 1) % SAMPLE_INTERVAL == 0:

            generation_model = get_raw_model(model)

            print("\n--- Generated Samples ---")

            generate_samples(
                generation_model,
                device,
            )

            print("\n--- End Generated Samples ---")

        if (step + 1) % SAVE_INTERVAL == 0:

            if distributed:
                dist.barrier()

            if is_master:
                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    step + 1,
                )
                print(f"Checkpoint saved: {checkpoint_path}")

            if distributed:
                dist.barrier()

    if MAX_STEPS > start_step and MAX_STEPS % SAVE_INTERVAL != 0:

        if distributed:
            dist.barrier()

        if is_master:
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                scaler,
                MAX_STEPS,
            )
            print(f"Final checkpoint saved: {checkpoint_path}")

        if distributed:
            dist.barrier()

    if is_master and MAX_STEPS % SAMPLE_INTERVAL != 0:

        generation_model = get_raw_model(model)

        print("\n--- Final Generated Samples ---")

        generate_samples(
            generation_model,
            device,
        )

        print("\n--- End Final Generated Samples ---")

    if MAX_STEPS > start_step and MAX_STEPS % HELLASWAG_EVAL_INTERVAL != 0:

        results = hellaswag.evaluate(
            model,
            split="val",
            device=device,
            max_examples=HELLASWAG_EVAL_EXAMPLES,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

        if is_master:
            print(
                f"Final HellaSwag accuracy: "
                f"{results['accuracy'] * 100:.2f}% "
                f"({results['correct']}/{results['total']})"
            )

    if distributed:
        dist.barrier()

    if is_master:
        print("Training complete.")


if __name__ == "__main__":
    try:
        train()
    finally:
        cleanup_distributed()
