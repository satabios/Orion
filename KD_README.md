# ORION Knowledge Distillation with 50% Head Pruning

This implementation provides Knowledge Distillation (KD) training for ORION with EVA-ViT head pruning, reducing the number of attention heads from 16 to 8 (50% pruning) while maintaining performance through teacher-student training.

## Overview

### Architecture Changes
- **Student Model**: EVA-ViT with 8 attention heads (reduced from 16)
- **Teacher Model**: Original ORION with full 16 attention heads  
- **Head Selection**: Magnitude-based importance scoring
- **Weight Transfer**: Intelligent transfer of selected head weights

### Knowledge Distillation Strategy
- **Feature Distillation**: Match intermediate feature representations
- **Attention Distillation**: Transfer attention patterns from selected heads
- **Output Distillation**: Soft target matching for final predictions
- **Scheduled Training**: Higher distillation weight during warmup

## Implementation Files

### Core Components
```
mmcv/models/backbones/eva_vit_student.py     # Pruned EVA-ViT backbone
mmcv/models/detectors/orion_student.py       # Student detector model
mmcv/models/losses/kd_losses.py              # Distillation loss functions
adzoo/orion/configs/orion_stage3_kd_train.py # KD training configuration
adzoo/orion/apis/train_kd.py                 # KD training loop
tools/head_pruning_utils.py                  # Head importance analysis
```

### Configuration Files
```
adzoo/orion/configs/orion_stage3_kd_train.py # Main KD training config
demo_kd_50_pruning.py                        # Demo script
```

## Quick Start

### Prerequisites
1. **Teacher Model**: Fully trained ORION model at `ckpt/orion/orion.pth`
2. **Dataset**: Chat-B2D dataset in `data/Chat-B2D/`
3. **Environment**: Same as original ORION requirements

### 1. Verify Setup
```bash
cd /Users/sathya/Desktop/Projects/Orion
python demo_kd_50_pruning.py --check-setup
```

### 2. Start KD Training
```bash
./adzoo/orion/orion_dist_train.sh adzoo/orion/configs/orion_stage3_kd_train.py 8
```

### 3. Monitor Training
The training will show both task losses and distillation losses:
```
Iter [100/1830] lr: 9.995e-05, eta: 2:15:48, time: 4.683, data_time: 0.171, memory: 12594, 
loss_cls: 0.8234, loss_bbox: 0.2156, distill_loss: 0.1876, total_loss: 0.9123
```

## Configuration Details

### KD Parameters
```python
kd_config = dict(
    teacher_model_path='ckpt/orion/orion.pth',     # Teacher checkpoint
    distillation_alpha=0.4,                        # KD loss weight
    distillation_temperature=4.0,                  # Soft target temperature
    feature_distill_weight=0.3,                    # Feature distillation
    attention_distill_weight=0.2,                  # Attention distillation
    output_distill_weight=0.3,                     # Output distillation
    warmup_epochs=2,                               # Warmup period
    warmup_alpha=0.6,                              # Higher KD weight during warmup
    final_alpha=0.3,                               # Final KD weight
    pruning_ratio=0.5,                             # 50% head reduction
)
```

### Model Configuration
```python
model = dict(
    type='OrionStudent',                           # Use student model
    img_backbone=dict(
        type='EVAViTStudent',                      # Pruned backbone
        num_heads=8,                               # Student heads
        teacher_num_heads=16,                      # Teacher heads
        # ... other params same as original
    ),
    distillation_loss=dict(
        type='CombinedKDLoss',                     # Combined KD losses
        # ... loss weights
    ),
    # ... rest same as original ORION
)
```

## Head Importance Analysis

### Compute Head Importance
```python
from tools.head_pruning_utils import compute_head_importance_magnitude
from mmcv.models import build_model

# Load teacher model
teacher_model = build_model(teacher_config)
checkpoint = torch.load('ckpt/orion/orion.pth')
teacher_model.load_state_dict(checkpoint['state_dict'])

# Compute importance scores
importance = compute_head_importance_magnitude(
    teacher_model, 
    save_path='work_dirs/head_importance.pkl'
)

# Analyze pruning impact
from tools.head_pruning_utils import analyze_pruning_impact
analyze_pruning_impact(importance, pruning_ratios=[0.25, 0.5, 0.75])
```

### Available Importance Methods
1. **Magnitude-based**: L2 norm of QKV projection weights
2. **Gradient-based**: Accumulated gradient magnitudes during training
3. **Entropy-based**: Attention pattern entropy analysis

## Training Process

### Two-Phase Training
1. **Warmup Phase** (Epochs 1-2):
   - Higher distillation weight (α=0.6)
   - Lower learning rate
   - Focus on knowledge transfer

2. **Fine-tuning Phase** (Epochs 3-6):
   - Lower distillation weight (α=0.3)
   - Standard learning rate
   - Balance task performance and efficiency

### Loss Function
```python
total_loss = (1-α) * student_task_loss + α * distillation_loss

distillation_loss = λ₁ * feature_loss + λ₂ * attention_loss + λ₃ * output_loss
```

## Expected Results

### Performance Metrics
- **Model Size**: ~50% reduction in attention parameters
- **Inference Speed**: 20-30% faster
- **Memory Usage**: 25-40% reduction
- **Task Performance**: <5% degradation expected

### Training Logs
Monitor these key metrics during training:
- `student_task_loss`: Original ORION losses
- `feature_distill_loss`: Feature-level distillation
- `attention_distill_loss`: Attention pattern transfer
- `output_distill_loss`: Soft target matching
- `total_distill_loss`: Combined distillation loss

## Advanced Usage

### Custom Pruning Ratios
To use different pruning ratios, modify the config:
```python
# For 25% pruning (keep 12 heads)
img_backbone.num_heads = 12
kd_config.pruning_ratio = 0.75

# For 75% pruning (keep 4 heads)  
img_backbone.num_heads = 4
kd_config.pruning_ratio = 0.25
```

### Custom Head Selection
Pre-compute head selections and pass to model:
```python
# Compute custom head importance
importance = compute_head_importance_gradient(teacher_model, dataloader)

# Create pruning config
from tools.head_pruning_utils import create_pruning_config
pruning_config = create_pruning_config(importance, pruning_ratio=0.5)

# Pass to model config
model.img_backbone.selected_heads_per_layer = pruning_config
```

### Different Distillation Strategies
Customize distillation loss weights:
```python
distillation_loss = dict(
    type='CombinedKDLoss',
    feature_distill_weight=0.5,    # Emphasize features
    attention_distill_weight=0.1,  # Reduce attention distillation
    output_distill_weight=0.3,     # Standard output distillation
    vl_distill_weight=0.1,         # Light VL distillation
)
```

## Troubleshooting

### Common Issues

1. **Teacher Model Not Found**
   ```
   Error: Teacher model not found at ckpt/orion/orion.pth
   Solution: Ensure teacher checkpoint exists and path is correct
   ```

2. **Out of Memory**
   ```
   Error: CUDA out of memory
   Solution: Reduce batch size or use gradient checkpointing
   ```

3. **Distillation Loss Not Decreasing**
   ```
   Check: Temperature parameter and loss weights
   Solution: Adjust distillation_alpha and temperature
   ```

### Debugging Tips

1. **Verify Head Transfer**:
   ```python
   # Check if heads were transferred correctly
   student_model.img_backbone.blocks[0].attn.selected_heads
   ```

2. **Monitor Loss Components**:
   ```python
   # Enable detailed logging in config
   log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook", by_epoch=True)])
   ```

3. **Validate Head Importance**:
   ```python
   # Save and visualize head importance scores
   importance = compute_head_importance_magnitude(teacher_model, save_path='debug/importance.pkl')
   ```

## Citation

If you use this KD implementation, please cite:
```bibtex
@article{orion2024,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={...},
  journal={...},
  year={2024}
}
```

## Contact

For questions about the KD implementation, please open an issue or contact the maintainers.