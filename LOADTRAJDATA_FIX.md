# LoadTrajData Pipeline Error - Solution

## ❌ Error
```
KeyError: "B2DOrionDataset: 'LoadTrajData is not in the pipeline registry'"
```

## 🔍 Root Cause

The error occurs because `LoadTrajData` pipeline component **does not exist** in the ORION codebase. This was mistakenly included in an earlier version of the config.

## ✅ Solution

The issue has been **fixed in the local config file** at:
```
adzoo/orion/configs/orion_stage3_kd_train.py
```

However, you're running training on a **remote server** where the old config is still cached.

## 🚀 Steps to Fix on Training Server

### Step 1: Sync Updated Files to Server
```bash
# From your local machine, sync the project to the server
rsync -avz --progress /Users/sathya/Desktop/Projects/Orion/ \
    user@server:/local/mnt/workspace/users/sathya/projects/Orion-KD/Orion/
```

### Step 2: Clean Python Cache on Server
```bash
# SSH into the training server
ssh user@server

# Navigate to the project directory
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion

# Clean ALL Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo "✓ Cache cleaned!"
```

### Step 3: Verify Configuration
```bash
# On the server, verify the config is correct
python verify_config.py
```

Expected output:
```
✅ SUCCESS: No LoadTrajData in pipeline
   Configuration is correct!
```

### Step 4: Restart Training
```bash
# Start training with the corrected config
bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4
```

## 📋 What Was Changed

### ❌ OLD Pipeline (with error):
```python
train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='LoadTrajData'),  # ❌ THIS DOESN'T EXIST!
    dict(type='LoadMapData', ...),  # ❌ THIS ALSO REMOVED
    ...
]
```

### ✅ NEW Pipeline (corrected):
```python
train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True, with_light_state=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='LoadAnnoatationVQA', ...),  # VQA data loading
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys),
    dict(type='CustomCollect3D', keys=[...]),
]
```

## 🎯 Key Changes

1. **Removed `LoadTrajData`** - This component doesn't exist; trajectory data is loaded through the dataset
2. **Removed `LoadMapData`** - Not needed for backbone-only KD training
3. **Added `VADObjectRangeFilter` and `VADObjectNameFilter`** - Match original pipeline
4. **Added `ResizeCropFlipRotImage`** - Match original data augmentation
5. **Set `tokenizer=None`** - Disable LLM components for backbone-only training

## 📝 Files to Sync

Make sure these files are synced to your training server:

1. ✅ `adzoo/orion/configs/orion_stage3_kd_train.py` - Main config (FIXED)
2. ✅ `mmcv/models/backbones/eva_vit_student.py` - Student backbone
3. ✅ `mmcv/models/detectors/orion_student.py` - Student detector
4. ✅ `mmcv/models/losses/kd_losses.py` - KD losses
5. ✅ `tools/head_pruning_utils.py` - Pruning utilities
6. ✅ `verify_config.py` - Config verification script

## 🔧 Quick Fix Script

Create this script on your training server and run it:

```bash
#!/bin/bash
# quick_fix.sh

echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo "Verifying config..."
python verify_config.py

if [ $? -eq 0 ]; then
    echo "✅ Ready to train!"
    echo "Run: bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4"
else
    echo "❌ Config still has errors. Please sync files from local machine."
    exit 1
fi
```

## ⚠️ Important Notes

- **Cache is the enemy**: Python caches compiled `.pyc` files. Always clean cache after config changes!
- **Sync matters**: The local files are correct. Make sure they're synced to the server.
- **Verify first**: Always run `verify_config.py` before training to catch issues early.

## 🎉 After Fix

Once fixed, you should see:
```
Loading checkpoint from ckpts/orion/orion.pth
Transferring teacher weights to student backbone...
Starting training with 123,456,789 trainable parameters (backbone only)
```

No more `LoadTrajData` errors! 🚀
