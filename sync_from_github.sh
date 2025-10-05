#!/bin/bash
# Run this script ON YOUR TRAINING SERVER to sync the latest changes
# Server: autoaisys-lambda3.qualcomm.com
# Path: /local/mnt/workspace/users/sathya/projects/Orion-KD/Orion

echo "=========================================="
echo "Syncing Latest Changes from GitHub"
echo "=========================================="
echo ""

# Pull latest changes from GitHub
echo "Step 1/4: Pulling latest code from GitHub..."
echo "------------------------------------------"
git fetch origin
git pull origin main
if [ $? -ne 0 ]; then
    echo "❌ Git pull failed! Check for conflicts."
    exit 1
fi
echo "✓ Code synced from GitHub"
echo ""

# Clean Python cache
echo "Step 2/4: Cleaning Python cache..."
echo "------------------------------------------"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✓ Python cache cleaned"
echo ""

# Verify config
echo "Step 3/4: Verifying configuration..."
echo "------------------------------------------"
python verify_config.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Configuration verification FAILED!"
    exit 1
fi
echo ""

# Double-check for LoadTrajData
echo "Step 4/4: Double-checking for LoadTrajData..."
echo "------------------------------------------"
if grep -q "LoadTrajData" adzoo/orion/configs/orion_stage3_kd_train.py; then
    echo "❌ ERROR: LoadTrajData still found in config!"
    echo "   File not properly synced. Try:"
    echo "   1. Check git status"
    echo "   2. Run: git reset --hard origin/main"
    exit 1
else
    echo "✓ No LoadTrajData in config"
fi
echo ""

# Success!
echo "=========================================="
echo "✅ Server Ready for Training!"
echo "=========================================="
echo ""
echo "Start training with:"
echo "  bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4"
echo ""
