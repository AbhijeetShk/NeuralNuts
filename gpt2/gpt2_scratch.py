import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
import torch.nn as nn
import torch.nn.functional as F

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

def get_device(): 
        if torch.cuda.is_available(): 
            return torch.device("cuda") 
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps") 
        return torch.device("cpu")


device = get_device()
print("device:", device)

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


class GPT2MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()

        self.c_fc = nn.Linear(
            n_embd,
            4 * n_embd
        )

        self.c_proj = nn.Linear(
            4 * n_embd,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x
    
    
mlp = GPT2MLP(
    n_embd=config.n_embd
)

x = torch.randn(
    2,
    8,
    config.n_embd
)

y = mlp(x)

print("input :", x.shape)
print("output:", y.shape)


class GPT2Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        dropout=0.0,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(n_embd)

        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln_2 = LayerNorm(n_embd)

        self.mlp = GPT2MLP(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
    
    
block = GPT2Block(
    n_embd=config.n_embd,
    n_head=config.n_head,
    block_size=config.n_positions,
)

x = torch.randn(
    2,
    8,
    config.n_embd,
)

y = block(x)

print("input :", x.shape)
print("output:", y.shape)


for name, param in block.named_parameters():
    print(f"{name:30s} {tuple(param.shape)}")
    
    
class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embd,
        dropout=0.0,
    ):
        super().__init__()

        self.block_size = block_size

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "wpe": nn.Embedding(block_size, n_embd),
            "h": nn.ModuleList([
                GPT2Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]),
            "ln_f": LayerNorm(n_embd),
        })

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        # GPT-2 ties token embeddings and output projection weights.
        self.lm_head.weight = self.transformer["wte"].weight

    def forward(self, idx):
        B, T = idx.shape

        assert T <= self.block_size, (
            f"sequence length {T} exceeds block size "
            f"{self.block_size}"
        )

        tok_emb = self.transformer["wte"](idx)

        pos = torch.arange(
            T,
            device=idx.device,
        )

        pos_emb = self.transformer["wpe"](pos)

        x = tok_emb + pos_emb

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        logits = self.lm_head(x)

        return logits
    
class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embd,
        dropout=0.0,
    ):
        super().__init__()

        self.block_size = block_size

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "wpe": nn.Embedding(block_size, n_embd),
            "h": nn.ModuleList([
                GPT2Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]),
            "ln_f": LayerNorm(n_embd),
        })

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        # GPT-2 ties token embeddings and output projection weights.
        self.lm_head.weight = self.transformer["wte"].weight

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.block_size, (
            f"sequence length {T} exceeds block size "
            f"{self.block_size}"
        )

        tok_emb = self.transformer["wte"](idx)

        pos = torch.arange(
            T,
            device=idx.device,
        )

        pos_emb = self.transformer["wpe"](pos)

        x = tok_emb + pos_emb

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
    
            logits, _ = self(idx_cond)
    
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            topk_probs, topk_indices = torch.topk(
                probs,
                top_k,
                dim=-1,
            )

            ix = torch.multinomial(
                topk_probs,
                num_samples=1,
            )

            next_token = torch.gather(
                topk_indices,
                -1,
                ix,
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1,
            )
    
        return idx

model = GPT2(
    vocab_size=config.vocab_size,
    block_size=config.n_positions,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
)

idx = torch.randint(
    0,
    config.vocab_size,
    (2, 16),
)

logits, loss = model(idx)

print("input :", idx.shape)
print("logits:", logits.shape)
print("loss:", loss.item() if loss is not None else None)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"parameters: {num_params:,}")
print(f"parameters: {num_params / 1e6:.2f}M")

print(
    model.transformer["wte"].weight
    is model.lm_head.weight
)
# print(model.transformer.wte.weight is model.lm_head.weight) #model.transformer is Hugging Face’s GPT2Model, which exposes wte as an attribute, not a dictionary key. 

tokenizer = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)

print("vocab size:", tokenizer.vocab_size)

text = "Hello, my name is Abhijeet."

tokens = tokenizer.encode(text)

print("token ids:", tokens)
print("decoded:", tokenizer.decode(tokens))

idx = torch.tensor(
    [tokens],
    dtype=torch.long,
)

print("input shape:", idx.shape)

with torch.no_grad():
    logits, _ = model(idx)

print("logits shape:", logits.shape)

reference_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)

def load_gpt2_weights(model, reference_model):
    with torch.no_grad():

        # Token and positional embeddings
        model.transformer["wte"].weight.copy_(
            reference_model.transformer.wte.weight
        )

        model.transformer["wpe"].weight.copy_(
            reference_model.transformer.wpe.weight
        )

        # Transformer BLOCKS
        for i, block in enumerate(model.transformer["h"]):

            ref_block = reference_model.transformer.h[i]

            # LayerNorm 1
            block.ln_1.weight.copy_(
                ref_block.ln_1.weight
            )

            block.ln_1.bias.copy_(
                ref_block.ln_1.bias
            )

            # attention QKV
            block.attn.c_attn.weight.copy_(
                ref_block.attn.c_attn.weight.T
            )

            block.attn.c_attn.bias.copy_(
                ref_block.attn.c_attn.bias
            )

            # attention output projection
            block.attn.c_proj.weight.copy_(
                ref_block.attn.c_proj.weight.T
            )

            block.attn.c_proj.bias.copy_(
                ref_block.attn.c_proj.bias
            )

            # LayerNorm 2
            block.ln_2.weight.copy_(
                ref_block.ln_2.weight
            )

            block.ln_2.bias.copy_(
                ref_block.ln_2.bias
            )

            # MLP input projection
            block.mlp.c_fc.weight.copy_(
                ref_block.mlp.c_fc.weight.T
            )

            block.mlp.c_fc.bias.copy_(
                ref_block.mlp.c_fc.bias
            )

            # MLP output projection
            block.mlp.c_proj.weight.copy_(
                ref_block.mlp.c_proj.weight.T
            )

            block.mlp.c_proj.bias.copy_(
                ref_block.mlp.c_proj.bias
            )

        # last LayerNorm
        model.transformer["ln_f"].weight.copy_(
            reference_model.transformer.ln_f.weight
        )

        model.transformer["ln_f"].bias.copy_(
            reference_model.transformer.ln_f.bias
        )

    return model


model = GPT2(
    vocab_size=config.vocab_size,
    block_size=config.n_positions,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
)


load_gpt2_weights(
    model,
    reference_model,
)

print(
    torch.equal(
        model.transformer["wte"].weight,
        reference_model.transformer.wte.weight,
    )
)

print(
    torch.equal(
        model.transformer["ln_f"].weight,
        reference_model.transformer.ln_f.weight,
    )
)

print(
    torch.equal(
        model.transformer["h"][0].attn.c_attn.weight,
        reference_model.transformer.h[0].attn.c_attn.weight.T,
    )
)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"{num_params:,}")
print(f"{num_params / 1e6:.2f}M")


text = "The quick brown fox jumps over the lazy dog."

tokens = tokenizer.encode(text)

idx = torch.tensor(
    [tokens],
    dtype=torch.long,
)


reference_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)
reference_model.eval()

model = model.to(device)
model.eval()

idx_cpu = idx
idx_device = idx.to(device)

print("reference device:", next(reference_model.parameters()).device)
print("our model device:", next(model.parameters()).device)
print("input device:", idx_device.device)

with torch.no_grad():
    reference_logits = reference_model(idx_cpu).logits
    our_logits, _ = model(idx_device)

our_logits_cpu = our_logits.cpu()

print("reference:", reference_logits.shape)
print("ours     :", our_logits_cpu.shape)

diff = (reference_logits - our_logits_cpu).abs()

print("max diff :", diff.max().item())
print("mean diff:", diff.mean().item())
    
    
print("reference:", reference_logits.shape)
print("ours     :", our_logits.shape)

max_diff = (
    reference_logits - our_logits_cpu
).abs().max()

print("max absolute difference:", max_diff.item())

mean_diff = (
    reference_logits - our_logits_cpu
).abs().mean()

print("mean absolute difference:", mean_diff.item())


reference_next = torch.argmax(
    reference_logits[:, -1, :],
    dim=-1,
)

our_next = torch.argmax(
    our_logits[:, -1, :],
    dim=-1,
)

print("reference token:", reference_next.item())
print("our token      :", our_next.item())

print(
    "reference:",
    tokenizer.decode(reference_next.tolist())
)

print(
    "ours     :",
    tokenizer.decode(our_next.tolist())
)


k = 10

reference_top = torch.topk(
    reference_logits[:, -1, :],
    k=k,
    dim=-1,
)

our_top = torch.topk(
    our_logits[:, -1, :],
    k=k,
    dim=-1,
)

print("Reference:")
for token_id, score in zip(
    reference_top.indices[0],
    reference_top.values[0],
):
    print(
        repr(tokenizer.decode([token_id.item()])),
        score.item(),
    )

print("\nOurs:")
for token_id, score in zip(
    our_top.indices[0],
    our_top.values[0],
):
    print(
        repr(tokenizer.decode([token_id.item()])),
        score.item(),
    )
    
print(config.activation_function)

prompt = "The future of artificial intelligence"

tokens = tokenizer.encode(prompt)

idx = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device
)

generated = model.generate(
    idx,
    max_new_tokens=50,
)

text = tokenizer.decode(
    generated[0].tolist()
)

print(text)


#After adding topK
prompt = "The future of artificial intelligence"
torch.manual_seed(42)
idx = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device,
)

generated = model.generate(
    idx,
    max_new_tokens=50,
    top_k=50,
)

print(
    tokenizer.decode(
        generated[0].tolist()
    )
)

print("device:", device)
print("model:", next(model.parameters()).device)
print("input:", idx.device)



def get_batch(tokens, batch_size, block_size, device):
    max_start = len(tokens) - block_size - 1

    ix = torch.randint(
        max_start,
        (batch_size,),
    )

    x = torch.stack([
        tokens[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        tokens[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


model = GPT2(
    vocab_size=config.vocab_size,
    block_size=config.n_positions,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
)

load_gpt2_weights(model, reference_model)

model = model.to(device)
model.eval()

tokenizer = GPT2TokenizerFast.from_pretrained(
"openai-community/gpt2"
)


text = """
The future of artificial intelligence is uncertain.
Large language models learn statistical patterns from text.
Transformers use attention to model relationships between tokens.
"""

encoded = tokenizer.encode(text)

tokens = torch.tensor(
    encoded,
    dtype=torch.long,
)


x, y = get_batch(
    tokens,
    batch_size=4,
    block_size=16,
    device=device,
)

print(x.shape)
print(y.shape)

logits, loss = model(x)

print(logits.shape) #At each of 4 x 16 posn it produces 50257 scores. So total logit values = 4×16×50257

print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y[0].tolist()))

model = GPT2(
    vocab_size=config.vocab_size,
    block_size=config.n_positions,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
)

load_gpt2_weights(model, reference_model)

model = model.to(device)
model.eval()

idx = idx.cpu().to(device)

print("model device:", next(model.parameters()).device)
print("idx device:", idx.device)

with torch.no_grad():
    our_logits, _ = model(idx)

print("ours:", our_logits.shape)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

model.train()

for step in range(50):
    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(x, y)

    loss.backward()

    optimizer.step()

    if step % 5 == 0:
        print(f"step {step:02d} | loss {loss.item():.4f}")
        
        
model.eval()

with torch.no_grad():
    logits, loss = model(x, y)

print("final loss:", loss.item())