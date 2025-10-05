#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from mmcv import Config

def debug_config():
    """Debug what's in the model configuration."""
    try:
        cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_kd_train.py')
        print("✅ Config loaded successfully")
        print(f"Model type: {cfg.model.type}")
        
        # Check all model keys
        print("\n📋 Model keys:")
        for key in cfg.model.keys():
            print(f"  - {key}")
        
        # Specifically check for img_neck
        if hasattr(cfg.model, 'img_neck'):
            print(f"\n❌ img_neck found: {cfg.model.img_neck}")
        else:
            print("\n✅ img_neck not found")
            
        # Check the model structure
        print(f"\n🏗️ Model structure after img_backbone:")
        backbone_found = False
        for key in cfg.model.keys():
            if backbone_found:
                print(f"  Next component: {key}")
                break
            if key == 'img_backbone':
                backbone_found = True
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_config()