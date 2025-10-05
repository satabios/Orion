# 🚀 Quick Fix Guide: LoadTrajData Error

## Problem
```
KeyError: "B2DOrionDataset: 'LoadTrajData is not in the pipeline registry'"
```

## Cause
The config on your **training server** has cached Python files with the old (incorrect) pipeline that includes non-existent `LoadTrajData` component.

## Solution (3 Simple Steps)

### Step 1: Sync Files from Local to Server
```bash
# On your LOCAL machine (Mac)
cd /Users/sathya/Desktop/Projects/Orion

# Sync to training server (replace with your actual server details)
rsync -avz --progress ./ \
    username@autoaisys-lambda3.qualcomm.com:/local/mnt/workspace/users/sathya/projects/Orion-KD/Orion/

# Or use scp if rsync isn't available
scp -r ./* username@autoaisys-lambda3.qualcomm.com:/local/mnt/workspace/users/sathya/projects/Orion-KD/Orion/
```

### Step 2: SSH into Training Server and Run Fix Script
```bash
# SSH into your training server
ssh username@autoaisys-lambda3.qualcomm.com

# Navigate to project directory
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion

# Make fix script executable
chmod +x fix_training_server.sh

# Run the fix script
bash fix_training_server.sh
```

Expected output:
```
✅ Server Ready for Training!
✓ No LoadTrajData in pipeline
✓ Python cache cleaned
✓ Config verified
```

### Step 3: Start Training
```bash
# On the training server
bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4
```

## Alternative: Manual Fix

If you prefer to do it manually:

```bash
# SSH into server
ssh username@autoaisys-lambda3.qualcomm.com
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Verify config
python verify_config.py

# Start training
bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4
```

## What Was Fixed

The corrected pipeline in `adzoo/orion/configs/orion_stage3_kd_train.py` now matches the original ORION pipeline:

✅ **Removed**: `LoadTrajData` (doesn't exist)  
✅ **Removed**: `LoadMapData` (not needed)  
✅ **Added**: `VADObjectRangeFilter`, `VADObjectNameFilter`  
✅ **Added**: `ResizeCropFlipRotImage` for data augmentation  
✅ **Fixed**: `tokenizer=None` for backbone-only training  

## Files That Need to Be Synced

1. `adzoo/orion/configs/orion_stage3_kd_train.py` ⭐ **MOST IMPORTANT**
2. `mmcv/models/backbones/eva_vit_student.py`
3. `mmcv/models/detectors/orion_student.py`
4. `mmcv/models/losses/kd_losses.py`
5. `verify_config.py`
6. `fix_training_server.sh`

## Verification

After syncing and running the fix script, verify with:
```bash
python verify_config.py
```

Should show:
```
✅ SUCCESS: No LoadTrajData in pipeline
   Configuration is correct!
```

## Why This Happened

1. **Initial config** had `LoadTrajData` (mistake)
2. **Local files** were fixed
3. **Server files** weren't synced, so old cached `.pyc` files persisted
4. **Python** used cached files instead of reading the new config

## Prevention

Always after config changes:
```bash
# Clean cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Verify
python verify_config.py
```

---

**Ready to go! Just run the 3 steps above and training will start successfully.** 🚀
