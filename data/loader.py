"""
HuggingFace dataset loading + preprocessing for images & CadQuery code.

This module handles:
- Loading the CADCODER/GenCAD-Code dataset from HuggingFace
- Image preprocessing with vision transforms  
- CadQuery code tokenization with CodeT5
- Dataset preparation for PyTorch training

Usage:
    from data.loader import get_datasets
    train_ds, test_ds, tokenizer = get_datasets(subset=5000)
"""

import warnings
import os
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer

# Configuration constants
VISION_SIZE = 224                   # Input image size for ViT
MAX_LEN = 512                      # Maximum token length for code
CACHE_DIR = "./hf_cache"           # HuggingFace cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------ Vision transforms
vision_tfm = Compose([
    Resize((VISION_SIZE, VISION_SIZE)),  # Resize to standard ViT input size
    ToTensor(),                          # Convert PIL to tensor [0,1]
    Normalize([0.5]*3, [0.5]*3),        # Normalize to [-1, 1] range
])

# ---------------------------------------------------------------- Tokenizer setup
def build_tokenizer():
    """
    Build CodeT5 tokenizer with CadQuery-specific vocabulary expansion.
    
    Returns:
        AutoTokenizer: Configured tokenizer with additional CadQuery tokens
    """
    tok = AutoTokenizer.from_pretrained("Salesforce/codet5-small", use_fast=True)
    
    # Add common CadQuery patterns to vocabulary for better tokenization
    cadquery_tokens = ["cq.", "Workplane(", "box(", "extrude(", "cut(", 
                      "fillet(", "chamfer(", "hole(", "faces(", "edges("]
    tok.add_tokens(cadquery_tokens)
    
    return tok

# ---------------------------------------------------------------- Preprocessing
def make_preprocess(tokenizer):
    """
    Create preprocessing function for batched data processing.
    
    Args:
        tokenizer: The tokenizer to use for code encoding
        
    Returns:
        function: Preprocessing function for dataset.map()
    """
    def preprocess_batch(batch):
        # Transform images to tensor format for ViT encoder
        batch["pixel_values"] = [vision_tfm(img.convert("RGB")) for img in batch["image"]]
        
        # Tokenize CadQuery code for CodeT5 decoder
        encoding = tokenizer(
            batch["cadquery"],
            truncation=True,           # Truncate long sequences
            padding="max_length",      # Pad to consistent length
            max_length=MAX_LEN,        # Maximum sequence length
            return_tensors=None        # Return lists for batching
        )
        batch["input_ids"] = encoding["input_ids"]
        batch["attention_mask"] = encoding["attention_mask"]
        
        return batch
    
    return preprocess_batch

# ---------------------------------------------------------------- Public API
def get_datasets(subset=None, num_proc=8, streaming=False):
    """
    Load and preprocess the CadQuery dataset for training.
    
    Args:
        subset (int, optional): Limit dataset size for testing
        num_proc (int): Number of processes for data preprocessing
        streaming (bool): Whether to use streaming dataset
        
    Returns:
        tuple: (train_dataset, test_dataset, tokenizer)
    """
    # Define dataset splits
    split_train = f"train[:{subset}]" if subset else "train"
    split_test  = f"test[:{subset//5}]" if subset else "test"

    print(f"[loader] Loading splits: {split_train} | {split_test}")
    
    # Load datasets from HuggingFace Hub
    train_ds = load_dataset("CADCODER/GenCAD-Code",
                            split=split_train,
                            cache_dir=CACHE_DIR,
                            streaming=streaming)
    test_ds  = load_dataset("CADCODER/GenCAD-Code",
                            split=split_test,
                            cache_dir=CACHE_DIR,
                            streaming=streaming)

    # Initialize tokenizer and preprocessing function
    tokenizer = build_tokenizer()
    preprocess_fn = make_preprocess(tokenizer)

    # Remove original columns after preprocessing to save memory
    cols_drop = ["image", "cadquery", "deepcad_id",
                 "token_count", "prompt", "hundred_subset"]

    # Apply preprocessing with multiprocessing
    train_ds = train_ds.map(preprocess_fn, batched=True, num_proc=num_proc,
                            remove_columns=cols_drop)
    test_ds  = test_ds.map(preprocess_fn,  batched=True, num_proc=num_proc,
                            remove_columns=cols_drop)

    # Set format for PyTorch compatibility
    train_ds.set_format("torch",
                        columns=["pixel_values", "input_ids", "attention_mask"])
    test_ds.set_format("torch",
                       columns=["pixel_values", "input_ids", "attention_mask"])
    
    return train_ds, test_ds, tokenizer
    train_ds.set_format("torch",
                        columns=["pixel_values", "input_ids", "attention_mask"])
    test_ds.set_format("torch",
                       columns=["pixel_values", "input_ids", "attention_mask"])
    return train_ds, test_ds, tok
