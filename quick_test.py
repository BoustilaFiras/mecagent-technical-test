#!/usr/bin/env python3
"""
Quick validation script for Kaggle deployment
Tests core functionality without heavy computations
"""

def quick_validation():
    """Quick validation of core components"""
    print("🚀 QUICK VALIDATION FOR KAGGLE")
    print("=" * 50)
    
    # Test 1: Imports
    print("📦 Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        import cadquery as cq
        print(f"✅ Core libraries: PyTorch {torch.__version__}, Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Project modules
    print("🔧 Testing project modules...")
    try:
        import sys
        import os
        sys.path.append(os.getcwd())
        
        from data.loader import build_tokenizer
        from models.model_lora import build_model
        from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
        print("✅ Project modules imported successfully")
    except ImportError as e:
        print(f"❌ Project import error: {e}")
        return False
    
    # Test 3: Basic functionality
    print("⚡ Testing basic functionality...")
    try:
        # Test tokenizer
        tokenizer = build_tokenizer()
        print(f"✅ Tokenizer created (vocab size: {len(tokenizer)})")
        
        # Test metrics
        test_code = "import cadquery as cq\nresult = cq.Workplane('XY').box(10,10,10)"
        codes = {"test": test_code}
        vsr = evaluate_syntax_rate_simple(codes)
        print(f"✅ Metrics working (VSR: {vsr})")
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False
    
    print("\n🎉 VALIDATION PASSED!")
    print("✅ Ready for Kaggle training")
    return True

if __name__ == "__main__":
    import sys
    success = quick_validation()
    sys.exit(0 if success else 1)
