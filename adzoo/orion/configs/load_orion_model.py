#!/usr/bin/env python3
"""
A simple script to load the original ORION model with pre-trained weights.
"""

import os
import sys
import torch
from mmcv import Config
from mmcv.models import build_model
from mmcv.runner import load_checkpoint

# Add project root to Python path to allow local imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_orion_model(config_path, checkpoint_path):
    """
    Loads the ORION model from a config file and populates it with weights
    from a checkpoint file.

    Args:
        config_path (str): Path to the model configuration file.
        checkpoint_path (str): Path to the .pth weight file.

    Returns:
        torch.nn.Module: The loaded model.
    """
    print("=" * 80)
    print("Loading ORION Model")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint file not found at '{checkpoint_path}'")
        return None

    try:
        # 1. Load the configuration file
        cfg = Config.fromfile(config_path)
        print(f"✅ Configuration loaded from: {config_path}")

        # 2. Build the model architecture
        # We use the student config but ensure the model type is the original 'Orion'
        cfg.model.type = 'Orion'
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        print(f"✅ Model '{type(model).__name__}' built successfully.")

        # 3. Load the weights from the checkpoint
        load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False)
        print(f"✅ Weights loaded from: {checkpoint_path}")

        # 4. Set the model to evaluation mode
        model.eval()
        print("✅ Model set to evaluation mode.")
        print("=" * 80)
        return model

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # --- Configuration ---
    # Use the KD config as a base, as it contains all necessary components
    CONFIG_FILE = 'adzoo/orion/configs/orion_stage3_kd_train.py'
    CHECKPOINT_FILE = 'ckpts/orion/orion.pth'

    # --- Load the Model ---
    orion_model = load_orion_model(CONFIG_FILE, CHECKPOINT_FILE)

    if orion_model:
        print("\n🎉 ORION model is loaded and ready for inference!")
        # You can now use 'orion_model' for inference, analysis, etc.
        # For example, print the number of parameters:
        num_params = sum(p.numel() for p in orion_model.parameters())
        print(f"   - Total parameters: {num_params:,}")
