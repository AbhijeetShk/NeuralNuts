import torch
from transformers import GPT2LMHeadModel
import math

torch.manual_seed(1337)

model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)
config = model.config

print("Vocabulary size :", config.vocab_size)
print("Context length  :", config.n_positions)
print("Embedding dim   :", config.n_embd)
print("Layers          :", config.n_layer)
print("Attention heads :", config.n_head)

num_params = sum(p.numel() for p in model.parameters())

print(f"Total parameters: {num_params:,}")
print(f"Total parameters: {num_params / 1e6:.2f}M")

block = model.transformer.h[0]

print("\nToken embedding:", model.transformer.wte.weight.shape)
print("Position embedding:", model.transformer.wpe.weight.shape)
print("Attention QKV:", block.attn.c_attn.weight.shape)
print("Attention output:", block.attn.c_proj.weight.shape)
print("MLP first layer:", block.mlp.c_fc.weight.shape)
print("MLP second layer:", block.mlp.c_proj.weight.shape)


class GPT2Embeddings(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape

        positions = torch.arange(
            T,
            device=idx.device
        )

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(positions)

        return tok_emb + pos_emb
    
embeddings = GPT2Embeddings(
    vocab_size=config.vocab_size,
    block_size=config.n_positions,
    n_embd=config.n_embd,
)

idx = torch.randint(
    0,
    config.vocab_size,
    (2, 8),
)

x = embeddings(idx)

print("input shape :", idx.shape)
print("output shape:", x.shape)

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.bias = None

    def forward(self, x):
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )
        
ln = LayerNorm(config.n_embd)

x = torch.randn(2, 8, config.n_embd)
y = ln(x)

print("input :", x.shape)
print("output:", y.shape)
print("mean  :", y.mean().item())
print("std   :", y.std().item())


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # GPT-2 combines Q, K and V projections.
        self.c_attn = nn.Linear(
            n_embd,
            3 * n_embd
        )

        self.c_proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask.
        mask = torch.tril(
            torch.ones(block_size, block_size)
        )

        self.register_buffer(
            "bias",
            mask.view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # preventing attending to future tokens.
        att = att.masked_fill(
            self.bias[:, :, :T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # weighted combination of values.
        y = att @ v

        # [B, n_head, T, head_dim]
        # -> [B, T, n_head, head_dim]
        y = y.transpose(1, 2).contiguous()

        # merge heads.
        y = y.view(B, T, C)

        # output projection.
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y
    
attention = CausalSelfAttention(
    n_embd=config.n_embd,
    n_head=config.n_head,
    block_size=config.n_positions,
)

x = torch.randn(2, 8, config.n_embd)

y = attention(x)

print("input :", x.shape)
print("output:", y.shape)