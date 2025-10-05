#!/bin/bash
# Script to clean Python cache and ensure config is up-to-date

echo "=========================================="
echo "Cleaning Python Cache Files"
echo "=========================================="

# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "✓ Removed __pycache__ directories"

# Remove all .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✓ Removed .pyc files"

# Remove all .pyo files
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✓ Removed .pyo files"

echo ""
echo "=========================================="
echo "Verifying Configuration"
echo "=========================================="

# Verify the config is correct
python verify_config.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Configuration is valid!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Sync this directory to your training server"
    echo "2. On the server, run this script again to clean cache"
    echo "3. Then start training with:"
    echo "   bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4"
else
    echo ""
    echo "=========================================="
    echo "❌ Configuration has errors!"
    echo "=========================================="
    exit 1
fi
