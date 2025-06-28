"""
Evaluate valid syntax rate on test images using provided metrics
Usage:
python eval/valid_syntax.py --ckpt checkpoints/ce_run --n 200
"""
import argparse
import os
import json
import torch
from data.loader import get_datasets
from models.model_lora import build_model
# Use provided metrics from metrics/ folder
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best

def parse():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, 
                   help="Path to model checkpoint")
    p.add_argument("--n", type=int, default=200,
                   help="Number of test samples to evaluate")
    return p.parse_args()

def main():
    """Main evaluation function"""
    args = parse()
    
    # Load test dataset and model
    _, test_ds, tok = get_datasets(subset=args.n)
    model = build_model(tok).from_pretrained(args.ckpt).eval().cuda()

    # Generate codes for all test samples
    generated_codes = {}
    reference_codes = {}
    
    print(f"Evaluating {args.n} samples...")
    for i, sample in enumerate(test_ds):
        # Generate code from image
        with torch.no_grad():
            pred = model.generate(sample["pixel_values"].unsqueeze(0).cuda(),
                                  max_length=512)
        code = tok.decode(pred[0], skip_special_tokens=True)
        generated_codes[f"sample_{i}"] = code
        
        # Note: For full evaluation, we would need to load the original dataset
        # to get reference codes. Currently evaluating syntax only.
        
    # Evaluate syntax validity using provided metrics
    vsr = evaluate_syntax_rate_simple(generated_codes)
    
    print(f"Valid syntax rate: {vsr:.2%}")
    
    # Optional: Calculate IOU between first two generated codes as example
    if len(generated_codes) >= 2:
        codes_list = list(generated_codes.values())
        try:
            sample_iou = get_iou_best(codes_list[0], codes_list[1])
            print(f"Sample IOU between first two codes: {sample_iou:.3f}")
        except Exception as e:
            print(f"Could not calculate IOU: {e}")
    
    # Save results
    results = {
        "valid_rate": vsr,
        "num_samples": args.n,
        "generated_codes": generated_codes
    }
    
    with open("results_valid.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return vsr

if __name__ == "__main__":
    main()
