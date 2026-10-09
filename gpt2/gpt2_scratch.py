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

use_flash = device.type == "cuda"

print("flash attention:", use_flash)

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
                self.bias[:, :, :T, :T] == 0,
                float("-inf")
            )

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

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

def configure_optimizer(model, weight_decay, learning_rate):
    param_dict = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    decay_params = [
        param for param in param_dict.values()
        if param.dim() >= 2
    ]

    nodecay_params = [
        param for param in param_dict.values()
        if param.dim() < 2
    ]

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer

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
        
        self.act = nn.GELU(approximate="tanh")
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
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
    
# old but useful for reference    
# class GPT2(nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         block_size,
#         n_layer,
#         n_head,
#         n_embd,
#         dropout=0.0,
#     ):
#         super().__init__()

#         self.block_size = block_size

#         self.transformer = nn.ModuleDict({
#             "wte": nn.Embedding(vocab_size, n_embd),
#             "wpe": nn.Embedding(block_size, n_embd),
#             "h": nn.ModuleList([
#                 GPT2Block(
#                     n_embd=n_embd,
#                     n_head=n_head,
#                     block_size=block_size,
#                     dropout=dropout,
#                 )
#                 for _ in range(n_layer)
#             ]),
#             "ln_f": LayerNorm(n_embd),
#         })

#         self.lm_head = nn.Linear(
#             n_embd,
#             vocab_size,
#             bias=False,
#         )

#         # GPT-2 ties token embeddings and output projection weights.
#         self.lm_head.weight = self.transformer["wte"].weight

#     def forward(self, idx):
#         B, T = idx.shape

#         assert T <= self.block_size, (
#             f"sequence length {T} exceeds block size "
#             f"{self.block_size}"
#         )

#         tok_emb = self.transformer["wte"](idx)

#         pos = torch.arange(
#             T,
#             device=idx.device,
#         )

#         pos_emb = self.transformer["wpe"](pos)

#         x = tok_emb + pos_emb

#         for block in self.transformer["h"]:
#             x = block(x)

#         x = self.transformer["ln_f"](x)

#         logits = self.lm_head(x)

#         return logits
    
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

        # tying token embeddings and output projection weights.
        self.lm_head.weight = self.transformer["wte"].weight

        self.apply(self._init_weights)

        for block in self.transformer["h"]:
            torch.nn.init.normal_(
                block.attn.c_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layer),
            )

            torch.nn.init.normal_(
                block.mlp.c_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layer),
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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

original_vocab_size = config.vocab_size
padded_vocab_size = 50304
train_vocab_size = padded_vocab_size

print("original vocab:", original_vocab_size)
print("padded vocab:", padded_vocab_size)

def load_gpt2_weights(model, reference_model):
    
    with torch.no_grad():
        
        model.transformer["wte"].weight[:original_vocab_size].copy_(
        reference_model.transformer.wte.weight
        )
        
        # Token and positional embeddings
        model.transformer["wte"].weight[original_vocab_size:].normal_(
        mean=0.0,
        std=0.02,
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

model = model.to("cpu")
model = model.to(device)

idx = idx.to(device)

print("Model:", model.transformer["wte"].weight.device)
print("Input:", idx.device)


with torch.no_grad():
    reference_logits = reference_model(idx_cpu).logits
    our_logits, _ = model(idx_device)

with torch.no_grad():
    hf_logits = reference_model(idx.cpu()).logits
    our_logits, loss = model(idx)

# original GPT-2 vocabulary (50257 tokens)
our_original_logits = our_logits[..., :original_vocab_size]

# moving logits to CPU for comparing
hf_logits_cpu = hf_logits.cpu()
our_logits_cpu = our_original_logits.cpu()

print("HF logits shape:", hf_logits_cpu.shape)
print("Our logits shape:", our_logits_cpu.shape)

print(
    "max diff:",
    (hf_logits_cpu - our_logits_cpu).abs().max().item()
)

print(
    "mean diff:",
    (hf_logits_cpu - our_logits_cpu).abs().mean().item()
)

# predicting next token using only the original vocabulary
hf_next = torch.argmax(hf_logits_cpu[:, -1, :], dim=-1)
our_next = torch.argmax(our_logits_cpu[:, -1, :], dim=-1)

print("HF :", repr(tokenizer.decode(hf_next.tolist())))
print("Ours:", repr(tokenizer.decode(our_next.tolist())))



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


class DataLoaderLite:
    def __init__(self, B, T, tokenizer, text):
        self.B = B
        self.T = T
        self.tokenizer = tokenizer

        self.tokens = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long,
        )

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[
            self.current_position:
            self.current_position + B * T + 1
        ]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0

        return x.to(device), y.to(device)
    

train_text = """
The future of artificial intelligence is uncertain.
Artificial intelligence systems learn patterns from data.
Language models predict the next token given previous tokens.
Transformers use attention to process sequences of tokens.
Deep learning models improve through optimization.
""" * 100


B = 4
T = 16

train_loader = DataLoaderLite(
    B=B,
    T=T,
    tokenizer=tokenizer,
    text=train_text,
)

x, y = train_loader.next_batch()

print("x:", x.shape)
print("y:", y.shape)
print("x:")
print(x)

print("y:")
print(y)

print(torch.equal(x[:, 1:], y[:, :-1]))



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

model.train()

for step in range(20):
    x, y = train_loader.next_batch() #instead of roll using next_batch func
    
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


print(
    "tied embeddings:",
    model.transformer["wte"].weight.data_ptr()
    == model.lm_head.weight.data_ptr()
)


test_model = GPT2(
    vocab_size=config.vocab_size,
    block_size=config.n_positions,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
)

print("embedding std :", test_model.transformer["wte"].weight.std().item())
print("position std  :", test_model.transformer["wpe"].weight.std().item())
print("attn proj std :", test_model.transformer["h"][0].attn.c_proj.weight.std().item())
print("mlp proj std  :", test_model.transformer["h"][0].mlp.c_proj.weight.std().item())

# Select Device - On MPS so CUDA optimization needs some safety checks.

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    

print("device:", device)

if device.type == "cuda":
    amp_dtype = torch.bfloat16
    amp_enabled = True
else:
    amp_dtype = torch.float32
    amp_enabled = False

use_grad_scaler = (
    device.type == "cuda"
    and amp_dtype == torch.float16
)

scaler = (
    torch.amp.GradScaler("cuda")
    if use_grad_scaler
    else None
)

import time

train_model = GPT2(
    vocab_size=train_vocab_size,
    block_size=config.n_positions,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
)

# load_gpt2_weights(train_model, reference_model) dont need this cause i changed the vocab to fresh 50304 vocab size instead of 50257. So no need to load weights from reference model.

train_model = train_model.to(device)

if device.type == "cuda":
    train_model = torch.compile(train_model)
    
optimizer = torch.optim.AdamW(
    train_model.parameters(),
    lr=6e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

train_model.train()

for step in range(10):

    x, y = get_batch(
        tokens,
        batch_size=4,
        block_size=16,
        device=device,
    )

    start = time.perf_counter()

    logits, loss = train_model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        train_model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    print(
        f"step {step:02d} | "
        f"loss {loss.item():.4f} | "
        f"{elapsed * 1000:.2f} ms"
    )
    
    

def synchronize_device():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize() 

def benchmark_step(model, optimizer, x, y):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    synchronize_device()
    start = time.time()

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        logits, loss = model(x, y)

    synchronize_device()
    forward_time = time.time() - start

    start = time.time()

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    synchronize_device()
    backward_time = time.time() - start

    start = time.time()

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    synchronize_device()
    optimizer_time = time.time() - start

    return {
        "forward": forward_time,
        "backward": backward_time,
        "optimizer": optimizer_time,
        "loss": loss.item(),
    }
        
batch_size = 4
block_size = 16

x, y = get_batch(
    tokens,
    batch_size,
    block_size,
    device,
)

result = benchmark_step(
    train_model,
    optimizer,
    x,
    y,
)

print(f"forward : {result['forward']:.3f}s")
print(f"backward: {result['backward']:.3f}s")
print(f"optimizer: {result['optimizer']:.3f}s")
print(f"loss: {result['loss']:.4f}")

print(f"PyTorch: {torch.__version__}")
print(f"device: {device}")
print(f"AMP enabled: {amp_enabled}")
print(f"AMP dtype: {amp_dtype}")


# training loop with configured optimizer - weight decay and learning rate
optimizer = configure_optimizer(
    model,
    weight_decay=0.1,
    learning_rate=6e-4,
)
for i, group in enumerate(optimizer.param_groups):
    print(
        f"group {i}: "
        f"parameters={len(group['params'])}, "
        f"weight_decay={group['weight_decay']}"
    )
 
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 2
max_steps = 10

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)    
    
for step in range(max_steps):
    print(f"step {step:02d} | lr {get_lr(step):.6e}")
    
import time
model.train()

for step in range(max_steps):
    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch(
        tokens,
        batch_size=4,
        block_size=16,
        device=device,
    )

    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(x, y)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    print(
        f"step {step:02d} | "
        f"loss {loss.item():.4f} | "
        f"lr {lr:.2e} | "
        f"{elapsed * 1000:.2f} ms"
    )