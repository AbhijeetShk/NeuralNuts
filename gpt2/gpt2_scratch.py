import torch
from transformers import GPT2LMHeadModel

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