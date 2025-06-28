"""
ViT + GPT2 baseline model for Vision-to-Code generation
Simple architecture without LoRA for initial testing
"""
from transformers import VisionEncoderDecoderModel

def build_model(tokenizer, r=8, alpha=32):
    """
    Build Vision-to-Code model (baseline without LoRA for testing)
    
    Args:
        tokenizer: Tokenizer for the decoder
        r: LoRA rank (unused in baseline)
        alpha: LoRA alpha (unused in baseline)
    
    Returns:
        VisionEncoderDecoderModel ready for training
    """
    # Create vision-encoder-decoder model with compatible components
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k",     # Vision encoder
        "microsoft/DialoGPT-small"               # Text decoder (compatible)
    )

    # Configure the decoder for code generation
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id           = tokenizer.pad_token_id 
    model.config.eos_token_id           = tokenizer.eos_token_id

    # Gradient checkpointing to reduce VRAM usage
    model.encoder.gradient_checkpointing_enable()
    model.decoder.gradient_checkpointing_enable()
    
    print(f"✅ Model created successfully (Encoder: ViT, Decoder: GPT2)")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model params: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    return model
