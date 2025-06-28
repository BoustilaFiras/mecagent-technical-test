"""
CadQuery code generation from images using trained vision-to-code model

This script performs inference on a single image to generate CadQuery code.
The generated code can be executed to create 3D CAD models.

Usage:
    python infer.py --img sample.png --ckpt checkpoints/ce_run --out generated_code.py
"""
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer
from models.model_lora import build_model

# Vision preprocessing pipeline (same as training)
vision_transform = Compose([
    Resize((224, 224)),                    # Resize to ViT input size
    ToTensor(),                            # Convert to tensor [0,1]
    Normalize([0.5]*3, [0.5]*3)           # Normalize to [-1, 1]
])

def parse():
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Generate CadQuery code from image")
    p.add_argument("--img", required=True, help="Input image path (PNG/JPG)")
    p.add_argument("--ckpt", required=True, help="Model checkpoint directory")
    p.add_argument("--out", default="generated_code.py", help="Output Python file")
    p.add_argument("--max_length", type=int, default=512, help="Maximum code length")
    return p.parse_args()

def main():
    """Main inference function"""
    args = parse()
    
    print(f"Loading model from {args.ckpt}...")
    # Load tokenizer and model from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = build_model(tokenizer).from_pretrained(args.ckpt).eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Processing image: {args.img}")
    # Load and preprocess image
    image = Image.open(args.img).convert("RGB")
    img_tensor = vision_transform(image).unsqueeze(0).to(device)

    print("Generating CadQuery code...")
    # Generate CadQuery code
    with torch.no_grad():
        generated_ids = model.generate(
            img_tensor, 
            max_length=args.max_length,
            num_beams=4,                   # Beam search for better quality
            temperature=0.7,               # Some randomness
            do_sample=True
        )
    
    # Decode generated tokens to text
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Save generated code
    with open(args.out, "w") as f:
        f.write(generated_code)
    
    print(f"✅ Generated code saved to {args.out}")
    print(f"Preview:\n{generated_code[:200]}...")
    
    # Decode and save generated code
    code = tok.decode(pred[0], skip_special_tokens=True)
    with open(a.out, "w") as f:
        f.write(code)
    
    print(f"✅ Code written to {a.out}")

if __name__ == "__main__":
    main()
