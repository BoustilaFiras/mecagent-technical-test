"""
Cross-Entropy training with label smoothing for CadQuery code generation
Usage:
    accelerate launch train/train_ce.py --subset 50000 \
        --per_device_train_batch_size 4 --gradient_accumulation_steps 4
"""
import argparse
import torch
from transformers import (Seq2SeqTrainingArguments, Seq2SeqTrainer)
from data.loader import get_datasets
from models.model_lora import build_model

def create_vision_collator(tokenizer):
    """Create custom collator for vision-encoder-decoder model"""
    def collate_fn(batch):
        # Extract pixel values (images) and input_ids (target code)
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        # Pad the sequences
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(padded_input_ids),
            'decoder_attention_mask': torch.tensor(padded_attention_mask)
        }
    
    return collate_fn

def parse():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=int, default=None, 
                   help="Dataset subset size (None for full dataset)")
    p.add_argument("--output_dir", type=str, default="checkpoints/ce_run",
                   help="Output directory for checkpoints")
    p.add_argument("--per_device_train_batch_size", type=int, default=1,
                   help="Batch size per device")
    p.add_argument("--gradient_accumulation_steps", type=int, default=16,
                   help="Gradient accumulation steps")
    p.add_argument("--num_train_epochs", type=float, default=1,
                   help="Number of training epochs (can be decimal)")
    return p.parse_args()

def main():
    """Main training function"""
    args = parse()
    
    # Load datasets and tokenizer
    train_ds, val_ds, tok = get_datasets(subset=args.subset)
    
    # Build model with LoRA
    model = build_model(tok)

    # Training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=5e-4,                 # Conservative learning rate
        warmup_steps=200,                   # Warmup for stability
        label_smoothing_factor=0.1,         # Label smoothing
        fp16=True,                          # Mixed precision
        eval_strategy="epoch",              # Fixed parameter name
        save_strategy="epoch",              # Save each epoch
        save_total_limit=1,                 # Keep only best checkpoint
        report_to="none",                   # Disable wandb/tensorboard
        logging_steps=100,                  # Log every 100 steps
    )

    # Data collator for vision-encoder-decoder
    collator = create_vision_collator(tok)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,  # Use processing_class instead of tokenizer
        data_collator=collator
    )
    
    # Start training
    trainer.train()
    
    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
