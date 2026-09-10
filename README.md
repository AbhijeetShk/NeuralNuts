# Neural Networks from Scratch

A from-scratch implementation and exploration of modern neural network and language-modeling systems, progressing from character-level language models to Transformer architectures and GPT tokenization.

The project focuses on understanding the mechanics behind these systems by implementing core components directly in PyTorch, validating them against autograd/reference implementations, and extending them with independent experiments.

## Implemented

### Character-Level Language Models
- Bigram language modeling and autoregressive sampling
- MLP language models with learned character embeddings
- Context-window based prediction
- Hierarchical / progressively structured context modeling
- Training, validation, and test evaluation
- Temperature-controlled generation
- Perplexity and loss analysis

### Optimization & Training Diagnostics
- Activation and gradient statistics
- Dead/saturated neuron diagnostics
- Batch Normalization
- Manual forward/backward derivations
- Manual implementation of:
  - Softmax
  - Cross-entropy
  - BatchNorm gradients
  - Parameter gradients
- Numerical validation of manual gradients against PyTorch autograd
- Train/validation/test evaluation and checkpointing

### Attention & Transformers
- Causal context aggregation
- Self-attention from first principles
- Scaled dot-product attention
- Learned Query / Key / Value projections
- Causal masking
- Attention dropout
- Reusable attention heads
- Multi-head self-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Positional embeddings
- Transformer blocks
- Autoregressive text generation

### GPT
- Character/byte-level language modeling pipeline
- Configurable datasets and batch loading
- Training and generation scripts
- Checkpoint save/resume
- Synthetic arithmetic reasoning experiments
- Causality tests to verify that future tokens cannot influence earlier positions

### Tokenization & BPE
- Unicode and UTF-8 exploration
- Byte-level text representation
- Byte Pair Encoding implemented from scratch
- Pair-frequency statistics and iterative merging
- Vocabulary construction
- BPE encoding and decoding
- Tokenizer serialization
- Regex-based pre-tokenization
- GPT-2 tokenizer exploration/reimplementation
- GPT-4-style tokenization and `cl100k_base` comparison
- Special-token handling
- Tokenization compression statistics
- Character vs byte vs BPE comparisons
- Multilingual / Unicode round-trip validation
- `tiktoken` compatibility testing
- SentencePiece exploration
- Prompt compression and learned-token concepts
- Multimodal tokenization and vector-quantized representations

## Validation & Experiments

Implementations are tested through:

- Shape and dimensionality checks
- Encode/decode round-trip tests
- Unicode and multilingual inputs
- Causal masking tests
- Numerical gradient checks against PyTorch autograd
- Reference comparisons against established tokenizers
- Compression-ratio measurements
- Temperature-based sampling experiments
- Train/validation/test loss and perplexity
- Synthetic reasoning datasets
- Edge-case and failure-mode exploration

## Philosophy

The objective is not to reproduce a framework or hide complexity behind abstractions.

Core mechanisms are rebuilt from first principles, inspected at the tensor level, tested against independent references, and extended beyond the original implementations where useful.

> Understand the mechanism → implement it → validate it → extend it.