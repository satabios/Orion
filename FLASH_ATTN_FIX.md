# Missing flash_attn Dependency Fix

## ❌ Error
```
ModuleNotFoundError: No module named 'flash_attn'
```

## ✅ Quick Fix

### Option 1: Install flash-attn (Recommended)
```bash
# On your training server
pip install flash-attn --no-build-isolation

# Or if you need a specific version
pip install flash-attn==2.3.3 --no-build-isolation
```

### Option 2: Disable Flash Attention (If installation fails)

If flash-attn installation fails (it requires specific CUDA/GPU setup), you can disable it:

#### Edit: `mmcv/models/utils/attention.py`

Change line 22 from:
```python
from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
```

To:
```python
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_unpadded_kvpacked_func = None
```

Then modify the FlashMHA class to check `HAS_FLASH_ATTN` before using it.

## 📋 Full Installation Command

```bash
# SSH into server
ssh username@autoaisys-lambda3.qualcomm.com

# Activate your conda environment
conda activate orion

# Install flash-attn
pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print('flash_attn installed:', flash_attn.__version__)"

# Start training
cd /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion
bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4
```

## ⚠️ Note

Flash Attention requires:
- NVIDIA GPU with Compute Capability >= 7.5 (Volta, Turing, Ampere, Ada, Hopper)
- CUDA 11.6+
- PyTorch 1.12+

If your environment doesn't support it, use Option 2 to disable it gracefully.
