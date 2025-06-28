# Vision-to-Code Generation for CadQuery

## Overview

This repository implements a baseline vision-encoder-decoder model for generating CadQuery code from 3D CAD images, addressing the challenge of automated parametric CAD modeling through deep learning.

## Architecture

**Model**: Vision Transformer (ViT-Base) encoder coupled with GPT2-Small decoder
- **Input**: 224×224 RGB images of 3D CAD models  
- **Output**: Executable CadQuery Python code
- **Training**: Cross-entropy loss with label smoothing (α=0.1)
- **Dataset**: CADCODER/GenCAD-Code (147K training pairs)

## Repository Structure

```
data/loader.py              # Dataset preprocessing and tokenization
models/model_lora.py        # VisionEncoderDecoderModel implementation  
train/
├── train_ce.py            # Cross-entropy training pipeline
└── train_rl.py            # PPO reinforcement learning
eval/metric_chamfer.py      # Geometric similarity evaluation
metrics/                    # Official evaluation framework
utils/mesh_utils.py         # STL processing and point cloud utilities
infer.py                    # Single-image inference script
kaggle_final_ready.ipynb    # Production deployment notebook
```

## Installation

```bash
pip install -r requirements.txt
python test_pipeline.py  # Validate installation
```

## Usage

**Training:**
```bash
PYTHONPATH=. accelerate launch train/train_ce.py --subset 50000
```

**Evaluation:**
```bash
python eval/metric_chamfer.py --ckpt checkpoints/ce_run --n 100
```

**Inference:**
```bash
python infer.py --img input.png --ckpt checkpoints/ce_run --out output.py
```
## Evaluation Metrics

**Primary Metrics:**
- **Valid Syntax Rate (VSR)**: Proportion of syntactically correct CadQuery code
- **Best IoU**: 3D geometric similarity via intersection-over-union
- **Chamfer Distance**: Point cloud geometric accuracy (lower is better)

## Performance Baselines

| Metric | Expected Range |
|--------|----------------|
| Valid Syntax Rate | 0.75 - 0.85 |
| Best IoU | 0.35 - 0.55 |
| Chamfer Distance | 0.45 - 0.65 |

## Technical Requirements

- **GPU**: 8GB+ VRAM (4GB minimum with batch_size=1)
- **Dependencies**: PyTorch ≥2.0, transformers ≥4.40, datasets ≥2.19
- **Training Time**: ~2 hours (50K samples, T4 GPU)

## Kaggle Deployment

Upload `kaggle_final_ready.ipynb` with GPU enabled (P100/T4 recommended).

---

*Implementation of vision-to-code generation for parametric CAD modeling*