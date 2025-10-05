#!/bin/bash
# Fix script for training server - Run this on the remote server after syncing files
# Usage: bash fix_training_server.sh

set -e  # Exit on error

echo "=========================================="
echo "ORION KD Training Server Fix Script"
echo "=========================================="
echo ""

# Step 1: Clean Python cache
echo "Step 1/4: Cleaning Python cache..."
echo "------------------------------------------"
find . -type d -name "__pycache__" -print -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -print -delete 2>/dev/null || true
find . -type f -name "*.pyo" -print -delete 2>/dev/null || true
echo "✓ Python cache cleaned"
echo ""

# Step 2: Verify Python environment
echo "Step 2/4: Verifying Python environment..."
echo "------------------------------------------"
python --version
echo "Python path: $(which python)"
echo ""

# Step 3: Verify configuration
echo "Step 3/4: Verifying configuration..."
echo "------------------------------------------"
if [ ! -f "verify_config.py" ]; then
    echo "❌ ERROR: verify_config.py not found!"
    echo "   Please sync all files from local machine first."
    exit 1
fi

python verify_config.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Configuration verification FAILED!"
    echo "   Please check the output above and fix errors."
    exit 1
fi
echo ""

# Step 4: Check checkpoint
echo "Step 4/4: Checking teacher checkpoint..."
echo "------------------------------------------"
if [ -f "ckpts/orion/orion.pth" ]; then
    SIZE=$(du -h ckpts/orion/orion.pth | cut -f1)
    echo "✓ Teacher checkpoint found: ckpts/orion/orion.pth ($SIZE)"
else
    echo "⚠️  WARNING: Teacher checkpoint not found at ckpts/orion/orion.pth"
    echo "   Training will use random initialization unless you provide the checkpoint."
fi
echo ""

# Final summary
echo "=========================================="
echo "✅ Server Ready for Training!"
echo "=========================================="
echo ""
echo "Configuration: adzoo/orion/configs/orion_stage3_kd_train.py"
echo "✓ No LoadTrajData in pipeline"
echo "✓ Python cache cleaned"
echo "✓ Config verified"
echo ""
echo "To start training, run:"
echo "  bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4"
echo ""
echo "Or for single GPU:"
echo "  python adzoo/orion/train.py adzoo/orion/configs/orion_stage3_kd_train.py --gpus 1"
echo ""
echo "=========================================="
