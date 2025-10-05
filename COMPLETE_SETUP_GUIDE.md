# 🚀 Complete Server Setup Guide

## Current Status
✅ **LoadTrajData error**: FIXED  
⚠️ **flash_attn missing**: Need to install

## Setup Steps (Run on Training Server)

### Step 1: Pull Latest Code
```bash
ssh username@autoaisys-lambda3.qualcomm.com
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion

# Pull from GitHub
git pull origin main
```

### Step 2: Activate Conda Environment
```bash
# Activate your conda environment
conda activate orion  # or whatever your environment name is

# Verify Python version
python --version  # Should be 3.8+
```

### Step 3: Install flash-attn
```bash
# This is the missing dependency
pip install flash-attn --no-build-isolation
```

**Note**: This may take 5-10 minutes to compile. It requires:
- CUDA 11.6+
- GPU with Compute Capability >= 7.5 (Volta, Turing, Ampere, Ada, Hopper)
- ~2GB disk space for build

### Step 4: Run Automated Setup Script
```bash
# This script does everything: clean cache, verify config, check dependencies
bash setup_training_environment.sh
```

Expected output:
```
✅ Environment Setup Complete!
  ✓ Python environment: orion
  ✓ flash-attn installed
  ✓ Python cache cleaned
  ✓ Configuration verified
```

### Step 5: Start Training
```bash
# 4 GPU training
bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4

# Or single GPU for testing
python adzoo/orion/train.py adzoo/orion/configs/orion_stage3_kd_train.py --gpus 1
```

## Manual Installation (If Script Fails)

### Install flash-attn manually:
```bash
# Option 1: From PyPI (recommended)
pip install flash-attn --no-build-isolation

# Option 2: From source (if PyPI fails)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

# Option 3: Specific version
pip install flash-attn==2.3.3 --no-build-isolation
```

### Verify installation:
```bash
python -c "import flash_attn; print('Version:', flash_attn.__version__)"
```

### Clean cache manually:
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```

## Troubleshooting

### Issue: flash-attn installation fails

**Symptom:**
```
ERROR: Could not build wheels for flash-attn
```

**Solutions:**

1. **Check CUDA version**:
   ```bash
   nvcc --version
   # Should be 11.6 or higher
   ```

2. **Check GPU**:
   ```bash
   nvidia-smi
   # Check your GPU model
   ```

3. **Install build dependencies**:
   ```bash
   pip install packaging ninja
   conda install -c conda-forge cudatoolkit-dev
   ```

4. **Try pre-built wheels**:
   ```bash
   # For CUDA 11.8
   pip install flash-attn --extra-index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install flash-attn --extra-index-url https://download.pytorch.org/whl/cu121
   ```

### Issue: Still getting LoadTrajData error

**Solution:**
```bash
# Make sure you pulled the latest code
git pull origin main

# Force clean
git clean -fdx
git pull origin main

# Clean cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```

### Issue: Out of memory during flash-attn installation

**Solution:**
```bash
# Use less parallel jobs
MAX_JOBS=2 pip install flash-attn --no-build-isolation

# Or build in /tmp (if home directory is small)
TMPDIR=/tmp pip install flash-attn --no-build-isolation
```

## Verification Checklist

Before training, verify:

- [ ] Git pulled latest code: `git log -1 --oneline`
- [ ] Conda environment activated: `echo $CONDA_DEFAULT_ENV`
- [ ] flash-attn installed: `python -c "import flash_attn"`
- [ ] No LoadTrajData in config: `grep LoadTrajData adzoo/orion/configs/orion_stage3_kd_train.py` (should be empty)
- [ ] Cache cleaned: `find . -name "*.pyc" | wc -l` (should be 0)
- [ ] Teacher checkpoint exists: `ls -lh ckpts/orion/orion.pth`

## Quick Commands Reference

```bash
# Pull and setup
git pull origin main
bash setup_training_environment.sh

# Start training
bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4

# Monitor training (in another terminal)
tail -f work_dirs/orion_stage3_kd_train/*/log.txt

# Check GPU usage
watch -n 1 nvidia-smi
```

## Expected Training Output

Once everything is working, you should see:
```
Loading checkpoint from ckpts/orion/orion.pth...
Initializing OrionStudent model...
Transferring teacher weights to student backbone...
✓ Teacher weights loaded successfully
✓ Non-backbone components frozen
Total parameters: ~350M
Trainable parameters: ~150M (backbone only)
Starting training...
Epoch 1, Iter 1: loss=2.345, backbone_distill_loss=0.567, lr=0.0002
```

---

**You're almost there! Just install flash-attn and you're good to go! 🚀**
