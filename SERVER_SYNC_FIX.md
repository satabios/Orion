# 🚨 IMMEDIATE FIX: LoadTrajData Error

## Problem
Your **training server** has the OLD config file with `LoadTrajData`.  
Your **local Mac** has the FIXED config file (no `LoadTrajData`).

## Root Cause
You pushed to GitHub with `./gitpush.sh`, but **haven't pulled on the server yet**.

## Solution (Run on Training Server)

### Option 1: Pull from GitHub (Recommended)
```bash
# SSH into your training server
ssh username@autoaisys-lambda3.qualcomm.com

# Navigate to project
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion

# Pull latest changes
git pull origin main

# Clean cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Verify fix
grep -q "LoadTrajData" adzoo/orion/configs/orion_stage3_kd_train.py && echo "❌ Still has LoadTrajData" || echo "✅ Fixed!"

# Start training
bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4
```

### Option 2: Use Automated Script
```bash
# On training server
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion
bash sync_from_github.sh
```

### Option 3: Manual Sync (if git isn't working)
```bash
# On your Mac - create a tarball
cd /Users/sathya/Desktop/Projects/Orion
tar -czf orion_kd_fixed.tar.gz adzoo/orion/configs/orion_stage3_kd_train.py \
    mmcv/models/backbones/eva_vit_student.py \
    mmcv/models/detectors/orion_student.py \
    mmcv/models/losses/kd_losses.py \
    verify_config.py

# Copy to server
scp orion_kd_fixed.tar.gz username@autoaisys-lambda3.qualcomm.com:/tmp/

# On server - extract
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion
tar -xzf /tmp/orion_kd_fixed.tar.gz

# Clean cache and verify
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
python verify_config.py
```

## Verification Commands

### Check if config is fixed:
```bash
# Should return nothing (no LoadTrajData)
grep "LoadTrajData" adzoo/orion/configs/orion_stage3_kd_train.py

# Should show correct pipeline
head -20 adzoo/orion/configs/orion_stage3_kd_train.py
```

### Check git status:
```bash
git status
git log -1 --oneline
```

## What You Should See (Fixed Config)

The training pipeline should look like:
```python
train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type='LoadAnnotations3D', ...),
    dict(type='VADObjectRangeFilter', ...),
    dict(type='VADObjectNameFilter', ...),
    dict(type='LoadAnnoatationVQA', ...),
    # NO LoadTrajData HERE!
    dict(type='ResizeCropFlipRotImage', ...),
    ...
]
```

## Quick Test
```bash
# On server, this should pass:
python verify_config.py

# Expected output:
# ✅ SUCCESS: No LoadTrajData in pipeline
#    Configuration is correct!
```

## Still Having Issues?

If `git pull` shows conflicts or doesn't update:
```bash
# Hard reset to latest GitHub version
git fetch origin
git reset --hard origin/main

# Then clean cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```

---

**Bottom Line**: Your local files are correct. Just need to sync them to the server!
