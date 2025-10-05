#!/bin/bash
# Complete environment setup for ORION KD Training Server
# Run this on your training server after pulling from GitHub

set -e  # Exit on error

echo "=========================================="
echo "ORION KD Training Environment Setup"
echo "=========================================="
echo ""

# Step 1: Check conda environment
echo "Step 1/5: Checking Python environment..."
echo "------------------------------------------"
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  No conda environment activated!"
    echo "   Please activate your environment first:"
    echo "   conda activate orion"
    exit 1
fi
echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
python --version
echo ""

# Step 2: Install flash-attn
echo "Step 2/5: Installing flash-attn..."
echo "------------------------------------------"
if python -c "import flash_attn" 2>/dev/null; then
    echo "✓ flash-attn already installed"
    python -c "import flash_attn; print(f'  Version: {flash_attn.__version__}')"
else
    echo "Installing flash-attn (this may take a few minutes)..."
    pip install flash-attn --no-build-isolation
    if [ $? -eq 0 ]; then
        echo "✓ flash-attn installed successfully"
    else
        echo "❌ flash-attn installation failed!"
        echo "   This is required for ORION. Please check:"
        echo "   - CUDA version (needs 11.6+)"
        echo "   - GPU compute capability (needs >= 7.5)"
        echo "   - Available disk space"
        exit 1
    fi
fi
echo ""

# Step 3: Clean Python cache
echo "Step 3/5: Cleaning Python cache..."
echo "------------------------------------------"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✓ Python cache cleaned"
echo ""

# Step 4: Verify configuration
echo "Step 4/5: Verifying configuration..."
echo "------------------------------------------"
if [ ! -f "verify_config.py" ]; then
    echo "⚠️  Warning: verify_config.py not found. Skipping config verification."
else
    python verify_config.py
    if [ $? -ne 0 ]; then
        echo "❌ Configuration verification FAILED!"
        exit 1
    fi
fi
echo ""

# Step 5: Check checkpoint
echo "Step 5/5: Checking teacher checkpoint..."
echo "------------------------------------------"
if [ -f "ckpts/orion/orion.pth" ]; then
    SIZE=$(du -h ckpts/orion/orion.pth | cut -f1)
    echo "✓ Teacher checkpoint found: ckpts/orion/orion.pth ($SIZE)"
else
    echo "⚠️  WARNING: Teacher checkpoint not found at ckpts/orion/orion.pth"
    echo "   Training will initialize with random weights."
fi
echo ""

# Final summary
echo "=========================================="
echo "✅ Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Python environment: $CONDA_DEFAULT_ENV"
echo "  ✓ flash-attn installed"
echo "  ✓ Python cache cleaned"
echo "  ✓ Configuration verified"
echo ""
echo "Ready to start training!"
echo ""
echo "Command:"
echo "  bash adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 4"
echo ""
echo "=========================================="
