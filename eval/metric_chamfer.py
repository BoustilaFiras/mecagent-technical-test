"""
Chamfer distance evaluation for CadQuery code generation models

This script evaluates model performance by:
1. Generating CadQuery code from test images
2. Converting both predicted and reference code to STL meshes
3. Sampling point clouds from meshes
4. Computing Chamfer distance between point clouds

Usage:
    python eval/metric_chamfer.py --ckpt checkpoints/ce_run --n 100
"""
import argparse
import cadquery as cq
import io
import json
import torch
import tempfile
import subprocess
import os
import numpy as np
from data.loader import get_datasets
from models.model_lora import build_model
from utils.mesh_utils import stl_to_points, chamfer_kdtree

def cadquery_to_stl(code: str) -> bytes | None:
    """
    Execute CadQuery code and export to STL bytes
    
    Args:
        code: CadQuery Python code string
    
    Returns:
        bytes: STL file content or None if failed
    """
    try:
        loc = {"cq": cq}
        exec(code, loc)  # Should produce a 'solid' object
        solid = loc.get("solid", None)
        if solid is None:
            return None
        
        # Export to STL format
        buff = io.BytesIO()
        cq.exporters.export(solid, buff, exportType="STL")
        return buff.getvalue()
    except Exception:
        return None

def parse():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    p.add_argument("--n", type=int, default=100, help="Number of test samples")
    return p.parse_args()

def main():
    """Main evaluation function"""
    a = parse()
    
    # Load test dataset and model
    _, test_ds, tok = get_datasets(subset=a.n)
    model = build_model(tok).from_pretrained(a.ckpt).eval().cuda()

    # Calculate Chamfer distances
    chamfers = []
    for sample in test_ds:
        # Generate code from image
        with torch.no_grad():
            out = model.generate(sample["pixel_values"].unsqueeze(0).cuda(),
                                 max_length=512)
        code = tok.decode(out[0], skip_special_tokens=True)
        
        # Convert both predicted and reference codes to STL
        stl_pred = cadquery_to_stl(code)
        stl_ref = cadquery_to_stl(tok.decode(sample["input_ids"],
                                             skip_special_tokens=True))
        
        # Calculate Chamfer distance if both STLs are valid
        if stl_pred and stl_ref:
            p = stl_to_points(stl_pred)
            q = stl_to_points(stl_ref)
            chamfers.append(chamfer_kdtree(p, q))
    
    # Calculate average Chamfer distance
    proxy = float(np.mean(chamfers)) if chamfers else 1.0
    print(f"Chamfer proxy ↓ (average): {proxy:.4f}")
    
    # Save results
    with open("results_chamfer.json", "w") as f:
        json.dump({"chamfer": proxy}, f)

if __name__ == "__main__":
    main()
