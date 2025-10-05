#!/usr/bin/env python
"""Verify the KD training config is correct"""

import sys
sys.path.insert(0, '.')

from mmcv import Config

# Load the config
cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_kd_train.py')

print("=" * 80)
print("Verifying ORION KD Training Configuration")
print("=" * 80)

# Check train pipeline
print("\n📋 Train Pipeline:")
for i, step in enumerate(cfg.train_pipeline, 1):
    print(f"  {i}. {step['type']}")

# Check test pipeline
print("\n📋 Test Pipeline:")
for i, step in enumerate(cfg.test_pipeline, 1):
    print(f"  {i}. {step['type']}")

# Check for LoadTrajData
has_load_traj = any('LoadTrajData' in str(step) for step in cfg.train_pipeline)
has_load_traj_test = any('LoadTrajData' in str(step) for step in cfg.test_pipeline)

print("\n" + "=" * 80)
if has_load_traj or has_load_traj_test:
    print("❌ ERROR: LoadTrajData found in pipeline!")
    print("   This component doesn't exist and will cause errors.")
    sys.exit(1)
else:
    print("✅ SUCCESS: No LoadTrajData in pipeline")
    print("   Configuration is correct!")

# Check KD config
print("\n📊 KD Configuration:")
print(f"  • Teacher path: {cfg.kd_config['teacher_backbone_path']}")
print(f"  • Distillation alpha: {cfg.kd_config['distillation_alpha']}")
print(f"  • Temperature: {cfg.kd_config['distillation_temperature']}")
print(f"  • Pruning ratio: {cfg.kd_config['pruning_ratio']}")

# Check model type
print(f"\n🤖 Model Type: {cfg.model['type']}")
print(f"📚 Dataset Type: {cfg.data['train']['type']}")

print("\n" + "=" * 80)
print("✅ Configuration verification complete!")
print("=" * 80)
