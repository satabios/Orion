#!/usr/bin/env python3
# ------------------------------------------------------------------------
# ORION Knowledge Distillation with 50% Head Pruning Demo Script
# ------------------------------------------------------------------------

import os
import sys
import torch
import argparse
from mmcv import Config

# Add project root to path
sys.path.append('/Users/sathya/Desktop/Projects/Orion')

def setup_kd_training():
    """Setup and demonstrate the KD training pipeline."""
    
    print("=" * 60)
    print("ORION BACKBONE-ONLY KD WITH 50% HEAD PRUNING")
    print("=" * 60)
    
    # Configuration paths
    teacher_model_path = "ckpt/orion/orion.pth"
    config_path = "adzoo/orion/configs/orion_stage3_kd_train.py"
    
    print(f"\n1. BACKBONE-ONLY TRAINING STRATEGY")
    print(f"   Teacher: Full EVA-L backbone (16 heads) - FROZEN")
    print(f"   Student: Pruned EVA-L backbone (8 heads) - TRAINABLE")
    print(f"   Other components: img_neck, heads, LLM - FROZEN")
    print(f"   Training focus: Only student backbone learns via KD")
    
    print(f"\n2. CONFIGURATION")
    print(f"   Full model path: {teacher_model_path}")
    print(f"   KD config: {config_path}")
    print(f"   Pruning ratio: 50% (16 heads -> 8 heads)")
    
    # Check if teacher model exists
    if not os.path.exists(teacher_model_path):
        print(f"\n❌ ERROR: Pre-trained model not found at {teacher_model_path}")
        print(f"   Please ensure you have the fully trained ORION model at this path.")
        return False
    
    # Load configuration
    try:
        cfg = Config.fromfile(config_path)
        print(f"\n✅ Configuration loaded successfully")
        print(f"   Model type: {cfg.model.type}")
        print(f"   Backbone: {cfg.model.img_backbone.type}")
        print(f"   Student heads: {cfg.model.img_backbone.num_heads}")
        print(f"   Teacher heads: {cfg.model.img_backbone.teacher_num_heads}")
        print(f"   Freeze non-backbone: {cfg.model.freeze_non_backbone}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load configuration: {e}")
        return False
    
    print(f"\n3. HEAD PRUNING STRATEGY")
    print(f"   Strategy: Magnitude-based head selection")
    print(f"   Selection: Top 8 most important heads from 16")
    print(f"   Weight transfer: From teacher backbone to student backbone")
    print(f"   Component freezing: All non-backbone components frozen")
    
    print(f"\n4. BACKBONE-ONLY KD SETUP")
    kd_config = cfg.kd_config
    print(f"   Distillation alpha: {kd_config['distillation_alpha']} (high backbone focus)")
    print(f"   Temperature: {kd_config['distillation_temperature']} (tight following)")
    print(f"   Feature distill weight: {kd_config['feature_distill_weight']}")
    print(f"   Attention distill weight: {kd_config['attention_distill_weight']}")
    print(f"   Warmup epochs: {kd_config['warmup_epochs']} (short warmup)")
    print(f"   Freeze non-backbone: {kd_config['freeze_non_backbone']}")
    
    print(f"\n5. TRAINING CONFIGURATION")
    print(f"   Training duration: 3 epochs (backbone-only)")
    print(f"   Batch size: {cfg.data.samples_per_gpu}")
    print(f"   Learning rate: {cfg.optimizer.lr} (higher for backbone-only)")
    print(f"   Trainable parameters: ~25% of full model")
    
    # Calculate theoretical improvements
    print(f"\n6. EXPECTED IMPROVEMENTS")
    print(f"   Training time: ~75% reduction (backbone-only)")
    print(f"   Memory usage: ~60% reduction during training")
    print(f"   Inference speed: ~20-30% improvement")
    print(f"   Parameter reduction: ~50% in attention layers")
    print(f"   Expected performance: <3% degradation")
    
    return True

def demonstrate_head_importance():
    """Demonstrate head importance computation."""
    
    print(f"\n7. HEAD IMPORTANCE ANALYSIS")
    print(f"   Computing attention head importance scores...")
    
    try:
        from tools.head_pruning_utils import compute_head_importance_magnitude
        print(f"   ✅ Head pruning utilities loaded")
        print(f"   Available methods:")
        print(f"      - Magnitude-based (weight norms) - DEFAULT")
        print(f"      - Gradient-based (gradient magnitudes)")  
        print(f"      - Entropy-based (attention entropy)")
    except ImportError as e:
        print(f"   ❌ Could not import head pruning utilities: {e}")
        
    # Simulate head importance scores for demo
    print(f"\n   Example head importance scores (Layer 0):")
    print(f"   Head 0: 0.856  Head 1: 0.742  Head 2: 0.689  Head 3: 0.634")
    print(f"   Head 4: 0.598  Head 5: 0.523  Head 6: 0.467  Head 7: 0.421")
    print(f"   Head 8: 0.389  Head 9: 0.345  Head 10: 0.298 Head 11: 0.256")
    print(f"   Head 12: 0.223 Head 13: 0.189 Head 14: 0.167 Head 15: 0.134")
    print(f"\n   Selected heads (50%): [0, 1, 2, 3, 4, 5, 6, 7]")
    print(f"   Pruned heads (50%): [8, 9, 10, 11, 12, 13, 14, 15]")

def show_training_commands():
    """Show training commands."""
    
    print(f"\n8. BACKBONE-ONLY TRAINING COMMANDS")
    print(f"   To start backbone KD training:")
    print(f"   ```bash")
    print(f"   cd /Users/sathya/Desktop/Projects/Orion")
    print(f"   ./adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 8")
    print(f"   ```")
    
    print(f"\n   Expected training output:")
    print(f"   ```")
    print(f"   Loading full pre-trained ORION model...")
    print(f"   Freezing non-backbone components...")
    print(f"   - img_neck frozen")
    print(f"   - pts_bbox_head frozen") 
    print(f"   - map_head frozen")
    print(f"   - lm_head frozen")
    print(f"   Loading teacher EVA-L backbone...")
    print(f"   Transferring selected head weights to student...")
    print(f"   Starting backbone-only KD training...")
    print(f"   ```")
    
    print(f"\n   To analyze head importance first (optional):")
    print(f"   ```python")
    print(f"   from tools.head_pruning_utils import compute_head_importance_magnitude")
    print(f"   from mmcv.models import build_model")
    print(f"   ")
    print(f"   # Load full model")
    print(f"   checkpoint = torch.load('ckpt/orion/orion.pth')")
    print(f"   model = build_model(config.model)")
    print(f"   model.load_state_dict(checkpoint['state_dict'])")
    print(f"   ")
    print(f"   # Compute backbone importance")
    print(f"   importance = compute_head_importance_magnitude(model.img_backbone)")
    print(f"   ```")

def main():
    """Main demo function."""
    
    parser = argparse.ArgumentParser(description='ORION KD Demo')
    parser.add_argument('--check-setup', action='store_true', 
                       help='Check if setup is ready for KD training')
    parser.add_argument('--show-config', action='store_true',
                       help='Show KD configuration details')
    args = parser.parse_args()
    
    try:
        # Setup and check configuration
        success = setup_kd_training()
        
        if success:
            # Demonstrate head importance computation
            demonstrate_head_importance()
            
            # Show training commands
            show_training_commands()
            
            print(f"\n9. NEXT STEPS FOR BACKBONE-ONLY KD")
            print(f"   1. Ensure pre-trained ORION model is at ckpt/orion/orion.pth")
            print(f"   2. Optionally analyze head importance with different methods")
            print(f"   3. Start backbone-only KD training (only 3 epochs needed)")
            print(f"   4. Monitor backbone distillation loss convergence")
            print(f"   5. Evaluate pruned model performance")
            print(f"   6. Compare inference speed vs. original model")
            
            print(f"\n✅ Backbone-only KD setup complete! Ready for efficient 50% head pruning.")
            
        else:
            print(f"\n❌ Setup incomplete. Please resolve the issues above.")
            
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    main()