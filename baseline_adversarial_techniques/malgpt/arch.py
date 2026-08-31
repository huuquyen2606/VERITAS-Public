# ============================================================================
# MalGPT ARCHITECTURE
# This model uses a Causal Language Model (GPT-2 architecture) to extract
# salient features from malware byte sequences and generate benign-looking
# payloads for single-shot evasion.
# ============================================================================

"""
MalGPT Architecture

This module defines the architectural setup for MalGPT. Since the paper
utilizes a Generative Pre-Trained Transformer (GPT-2) with 12 interconnected
decoder blocks, we leverage the HuggingFace `transformers` library.

The architecture consists of:
1. Custom Hex Tokenizer:
   - Treats raw binary files as text sequences.
   - Chunks hexadecimal strings into sets of 4 characters.
   - Uses a custom vocabulary size of 65,539 (16^4 hex combinations + 3 special tokens).

2. GPT-2 Causal Language Model (openai-community/gpt2):
   - 12 interconnected decoder blocks.
   - Self-attention mechanisms to learn long-range dependencies in byte sequences.
   - Token embeddings resized to match the custom hex vocabulary.

Key Paper:
- "Single-Shot Black-Box Adversarial Attacks Against Malware Detectors:
   A Causal Language Model Approach" (Hu et al., 2021)
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config


def create_hex_vocabulary():
    """
    Generates a complete dictionary of all possible 2-byte (4-hex-character)
    combinations, from '0000' to 'FFFF', plus necessary special tokens.
    """
    vocab = {}
    idx = 0

    # Generate 65,536 hex token combinations
    for i in range(0x10000):
        hex_token = f"{i:04X}"
        vocab[hex_token] = idx
        idx += 1

    # Add special tokens for transformer logic
    vocab["<PAD>"] = idx
    vocab["<BOS>"] = idx + 1
    vocab["<EOS>"] = idx + 2

    return vocab


class HexTokenizer:
    """
    Custom tokenizer designed to process hexadecimal representations of binary files.
    Matches the paper's requirement of delimiting sequences by sets of four characters.
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_token_id = vocab["<PAD>"]
        self.bos_token_id = vocab["<BOS>"]
        self.eos_token_id = vocab["<EOS>"]
        self.vocab_size = len(vocab)

    def __len__(self):
        return self.vocab_size

    def encode(self, text, max_length=None, padding=False, truncation=True):
        """Converts space-delimited hex strings into token IDs."""
        tokens = text.strip().split(" ")
        token_ids = [self.vocab[token] for token in tokens if token in self.vocab]

        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        if padding and max_length and len(token_ids) < max_length:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Converts token IDs back into space-delimited hex strings."""
        tokens = []
        for tid in token_ids:
            if torch.is_tensor(tid):
                tid = tid.item()
            token = self.inv_vocab.get(tid, "<UNK>")
            if skip_special_tokens and token in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def __call__(
        self, text, max_length=None, padding=False, truncation=True, return_tensors=None
    ):
        """Allows the tokenizer to be called like a standard HuggingFace tokenizer."""
        token_ids = self.encode(text, max_length, padding, truncation)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        return {"input_ids": token_ids}


def get_malgpt_model(vocab_size, pad_token_id, bos_token_id, eos_token_id):
    """
    Initializes the GPT-2 architecture configured specifically for the custom
    hex vocabulary.

    Returns:
        model: An un-trained GPT2LMHeadModel with 12 decoder layers and resized embeddings.
    """
    config = GPT2Config.from_pretrained("openai-community/gpt2")
    config.vocab_size = vocab_size
    config.pad_token_id = pad_token_id
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id

    # Initialize model with custom config
    model = GPT2LMHeadModel(config)

    # Resize token embeddings to exactly match our 65,539 vocabulary
    model.resize_token_embeddings(vocab_size)

    return model


__all__ = ["create_hex_vocabulary", "HexTokenizer", "get_malgpt_model"]
