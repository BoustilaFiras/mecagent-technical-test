"""
PPO Fine-tuning with geometric rewards
Reward = 1 / (1 + chamfer_distance) - 0.2 * (1 - syntax_valid)
Requires transformers-trl >= 0.8
"""
import argparse
import torch
import numpy as np
from trl import PPOTrainer, PPOConfig
from data.loader import get_datasets
from models.model_lora import build_model
from utils.mesh_utils import stl_to_points, chamfer_kdtree
import cadquery as cq
import io
import subprocess
import tempfile
import os

def cadquery_reward(code_pred: str, code_ref: str):
    """
    Calculate reward based on syntax validity and geometric similarity
    
    Args:
        code_pred: Generated CadQuery code
        code_ref: Reference CadQuery code
    
    Returns:
        float: Reward score (higher is better)
    """
    # 1. Check syntax validity
    try:
        loc = {"cq": cq}
        exec(code_pred, loc)
        solid = loc.get("solid")
        valid = 1.0 if solid else 0.0
    except Exception:
        return -0.2  # Penalty for syntax errors
    
    # 2. Calculate Chamfer distance proxy
    try:
        # Export predicted solid to STL
        buf_pred = io.BytesIO()
        cq.exporters.export(solid, buf_pred, exportType="STL")
        
        # Execute reference code and export
        exec(code_ref, {"cq": cq})
        solid_ref = locals().get("solid")
        buf_ref = io.BytesIO()
        cq.exporters.export(solid_ref, buf_ref, exportType="STL")
        
        # Calculate Chamfer distance
        p = stl_to_points(buf_pred.getvalue())
        q = stl_to_points(buf_ref.getvalue())
        cham = chamfer_kdtree(p, q)
        
        # Reward: high for low Chamfer + syntax validity bonus
        return 1.0 / (1 + cham) - 0.2 * (1 - valid)
    except Exception:
        return -0.2  # Penalty for geometry errors

def parse():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=int, default=10000,
                   help="Dataset subset size")
    p.add_argument("--steps", type=int, default=5000,
                   help="Number of PPO training steps")
    p.add_argument("--output_dir", type=str, default="checkpoints/ppo",
                   help="Output directory for PPO model")
    return p.parse_args()

def main():
    """Main PPO training function"""
    a = parse()
    
    # Load dataset and models
    tr, _, tok = get_datasets(subset=a.subset)
    model = build_model(tok).cuda()
    ref_model = build_model(tok).from_pretrained("checkpoints/ce_run").cuda()

    # PPO configuration
    cfg = PPOConfig(
        batch_size=1,
        mini_batch_size=1,
        optimize_cuda_cache=True,
        learning_rate=5e-5,         # Conservative LR for stability
        target_kl=0.2,              # KL divergence target
        ppo_epochs=4                # PPO update epochs
    )
    
    # Initialize PPO trainer
    trainer = PPOTrainer(cfg, model, ref_model, tok)

    # Training loop
    for step, sample in enumerate(tr):
        if step >= a.steps:
            break
            
        # Prepare inputs
        query_img = sample["pixel_values"].unsqueeze(0).cuda()
        ref_code = tok.decode(sample["input_ids"], skip_special_tokens=True)
        
        # Generate code with current model
        with torch.no_grad():
            gen_ids = trainer.model.generate(query_img, max_length=512)
        
        # Decode generated code
        code_pred = tok.decode(gen_ids[0], skip_special_tokens=True)
        
        # Calculate reward
        reward = cadquery_reward(code_pred, ref_code)
        
        # PPO update step
        trainer.step([gen_ids[0]], [reward])
        
        # Logging
        if (step + 1) % 100 == 0:
            print(f"[{step+1}/{a.steps}] reward avg : {reward:.3f}")
    
    # Save final model
    trainer.save_pretrained(a.output_dir)
    tok.save_pretrained(a.output_dir)

if __name__ == "__main__":
    main()
