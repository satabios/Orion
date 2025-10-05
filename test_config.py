#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from mmcv import Config
from mmcv.models import build_model

def test_config():
    """Test if the KD config can build the model successfully."""
    try:
        # Load the config
        cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_kd_train.py')
        print("✅ Config loaded successfully")
        
        # Test model building
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        print("✅ Model built successfully")
        print(f"Model type: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)