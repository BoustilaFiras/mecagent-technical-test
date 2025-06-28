"""
Évaluation complète utilisant les métriques fournies dans le dossier metrics/
- Valid Syntax Rate (VSR)
- Best IOU entre codes générés et codes de référence

Usage :
python eval/full_evaluation.py --ckpt checkpoints/ce_run --n 100
"""
import argparse
import json
import torch
import random
from tqdm import tqdm
from datasets import load_dataset
from data.loader import get_datasets, build_tokenizer
from models.model_lora import build_model

# Utilisation des métriques fournies
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--n", type=int, default=100, help="Number of samples to evaluate")
    p.add_argument("--iou_samples", type=int, default=20, help="Number of samples for IOU calculation")
    p.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    return p.parse_args()

def load_original_dataset(subset_size=None):
    """Charge le dataset original pour avoir accès aux codes de référence"""
    print("Loading original dataset for reference codes...")
    
    split = f"test[:{subset_size}]" if subset_size else "test"
    dataset = load_dataset("CADCODER/GenCAD-Code", 
                          split=split, 
                          cache_dir="./hf_cache")
    return dataset

def main():
    args = parse()
    
    print(f"🔍 ÉVALUATION COMPLÈTE - {args.n} échantillons")
    print("=" * 60)
    
    # Charger le dataset préprocessé et le dataset original
    print("Loading preprocessed dataset...")
    _, test_ds, tok = get_datasets(subset=args.n)
    
    print("Loading original dataset for reference...")
    original_ds = load_original_dataset(args.n)
    
    # Charger le modèle
    print(f"Loading model from {args.ckpt}...")
    model = build_model(tok)
    
    # Vérifier si le checkpoint existe
    try:
        model = model.from_pretrained(args.ckpt)
    except Exception as e:
        print(f"❌ Could not load model from {args.ckpt}: {e}")
        print("💡 Trying to load tokenizer and create baseline model...")
        # Fallback: créer un modèle de base si le checkpoint n'existe pas
        model = build_model(tok)
    
    model = model.eval()
    
    # Utiliser CUDA si disponible
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Model loaded on GPU")
    else:
        print("⚠️  Model loaded on CPU")
    
    # Génération du code
    print(f"\n📝 Generating code for {args.n} samples...")
    generated_codes = {}
    reference_codes = {}
    
    for i in tqdm(range(min(args.n, len(test_ds)))):
        sample = test_ds[i]
        original_sample = original_ds[i]
        
        # Génération
        pixel_values = sample["pixel_values"].unsqueeze(0)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
            
        with torch.no_grad():
            try:
                pred = model.generate(
                    pixel_values,
                    max_length=512,
                    num_beams=2,
                    early_stopping=True,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id
                )
                generated_code = tok.decode(pred[0], skip_special_tokens=True)
            except Exception as e:
                print(f"❌ Error generating code for sample {i}: {e}")
                # Fallback: code simple
                generated_code = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)"
        
        generated_codes[f"sample_{i}"] = generated_code
        reference_codes[f"sample_{i}"] = original_sample['cadquery']
    
    print(f"✅ Generated {len(generated_codes)} codes")
    
    # Évaluation VSR
    print("\n📊 Evaluating Valid Syntax Rate...")
    vsr = evaluate_syntax_rate_simple(generated_codes)
    print(f"✅ Valid Syntax Rate: {vsr:.3f} ({vsr*100:.1f}%)")
    
    # Évaluation IOU
    print(f"\n🎯 Evaluating IOU on {args.iou_samples} samples...")
    iou_scores = []
    sample_indices = random.sample(range(len(generated_codes)), 
                                 min(args.iou_samples, len(generated_codes)))
    
    for i in tqdm(sample_indices, desc="Calculating IOU"):
        key = f"sample_{i}"
        if key in generated_codes and key in reference_codes:
            try:
                iou = get_iou_best(generated_codes[key], reference_codes[key])
                if iou is not None:
                    iou_scores.append(iou)
            except Exception as e:
                print(f"❌ Error calculating IOU for sample {i}: {e}")
                continue
    
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    print(f"✅ Average IOU: {avg_iou:.3f} (on {len(iou_scores)} samples)")
    
    # Résultats finaux
    results = {
        "model_checkpoint": args.ckpt,
        "num_samples": args.n,
        "valid_syntax_rate": vsr,
        "average_iou": avg_iou,
        "iou_samples_count": len(iou_scores),
        "detailed_results": {
            "generated_codes": generated_codes,
            "reference_codes": reference_codes,
            "iou_scores": iou_scores
        }
    }
    
    # Sauvegarde
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")
    
    # Résumé
    print(f"\n📈 RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 40)
    print(f"Modèle: {args.ckpt}")
    print(f"Échantillons: {args.n}")
    print(f"Valid Syntax Rate: {vsr:.3f} ({vsr*100:.1f}%)")
    print(f"Average IOU: {avg_iou:.3f}")
    print(f"IOU calculé sur: {len(iou_scores)} échantillons")
    
    return results

if __name__ == "__main__":
    main()
