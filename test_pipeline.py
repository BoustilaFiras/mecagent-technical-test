#!/usr/bin/env python3
"""
Complete pipeline validation test

This script validates that all components work together:
1. Data loading and preprocessing
2. Model creation and basic functionality  
3. Official metrics evaluation
4. All utilities and modules

Run this before deploying to ensure everything is working.
"""

import sys
import traceback
from pathlib import Path

def test_data_loader():
    """Test data loading and preprocessing"""
    print("🔄 Testing data loader...")
    try:
        from data.loader import get_datasets
        train_ds, test_ds, tokenizer = get_datasets(subset=100)
        
        # Test data structure
        sample = train_ds[0]
        assert "pixel_values" in sample, "Missing pixel_values"
        assert "input_ids" in sample, "Missing input_ids"
        assert "attention_mask" in sample, "Missing attention_mask"
        
        print(f"✅ Data loader working - Train: {len(train_ds)}, Test: {len(test_ds)}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"❌ Data loader failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation with LoRA"""
    print("🔄 Testing model creation...")
    try:
        from data.loader import build_tokenizer
        from models.model_lora import build_model
        
        tokenizer = build_tokenizer()
        model = build_model(tokenizer)
        
        # Check model structure
        assert hasattr(model, 'encoder'), "Missing encoder"
        assert hasattr(model, 'decoder'), "Missing decoder"
        
        print("✅ Model creation working")
        print(f"   Encoder: {model.encoder.__class__.__name__}")
        print(f"   Decoder: {model.decoder.__class__.__name__}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_official_metrics():
    """Test official metrics from the metrics/ folder"""
    print("🔄 Testing official metrics...")
    try:
        from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
        from metrics.best_iou import get_iou_best
        
        # Test with simple valid code
        test_codes = {
            "simple_box": """
import cadquery as cq
result = cq.Workplane("XY").box(50, 30, 10)
""",
            "box_with_hole": """
import cadquery as cq
result = (
    cq.Workplane("XY")
    .box(50, 30, 10)
    .faces(">Z")
    .workplane()
    .hole(5)
)
"""
        }
        
        # Test Valid Syntax Rate
        vsr = evaluate_syntax_rate_simple(test_codes)
        assert 0.0 <= vsr <= 1.0, f"Invalid VSR: {vsr}"
        
        # Test IOU metric
        iou = get_iou_best(test_codes["simple_box"], test_codes["box_with_hole"])
        assert 0.0 <= iou <= 1.0, f"Invalid IOU: {iou}"
        
        print(f"✅ Official metrics working - VSR: {vsr:.3f}, IOU: {iou:.3f}")
        return True
    except Exception as e:
        print(f"❌ Official metrics failed: {e}")
        traceback.print_exc()
        return False

def test_mesh_utils():
    """Test mesh utilities"""
    print("🔄 Testing mesh utilities...")
    try:
        from utils.mesh_utils import stl_to_points, chamfer_kdtree
        import numpy as np
        
        # Create test point clouds
        p1 = np.random.rand(100, 3).astype(np.float32)
        p2 = np.random.rand(100, 3).astype(np.float32)
        
        # Test Chamfer distance
        chamfer_dist = chamfer_kdtree(p1, p2)
        assert isinstance(chamfer_dist, float), "Chamfer should return float"
        assert chamfer_dist >= 0, "Chamfer should be non-negative"
        
        print(f"✅ Mesh utilities working - Chamfer: {chamfer_dist:.3f}")
        return True
    except Exception as e:
        print(f"❌ Mesh utilities failed: {e}")
        traceback.print_exc()
        return False

def test_training_script():
    """Test training script imports and argument parsing"""
    print("🔄 Testing training script...")
    try:
        from train.train_ce import parse
        
        # Test argument parsing with defaults
        import sys
        old_argv = sys.argv
        sys.argv = ["train_ce.py"]  # Simulate script call
        
        args = parse()
        assert hasattr(args, 'subset'), "Missing subset argument"
        assert hasattr(args, 'output_dir'), "Missing output_dir argument"
        
        sys.argv = old_argv  # Restore
        
        print("✅ Training script working")
        return True
    except Exception as e:
        print(f"❌ Training script failed: {e}")
        traceback.print_exc()
        return False

def test_inference_script():
    """Test inference script imports"""
    print("🔄 Testing inference script...")
    try:
        from infer import vision_transform, parse
        
        # Test vision transform
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        transformed = vision_transform(dummy_img)
        
        assert transformed.shape == (3, 224, 224), f"Wrong transform shape: {transformed.shape}"
        
        print("✅ Inference script working")
        return True
    except Exception as e:
        print(f"❌ Inference script failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Pipeline Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Model Creation", test_model_creation), 
        ("Official Metrics", test_official_metrics),
        ("Mesh Utils", test_mesh_utils),
        ("Training Script", test_training_script),
        ("Inference Script", test_inference_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
