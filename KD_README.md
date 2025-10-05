# ORION Knowledge Distillation with 50% Head Pruning (Backbone-Only Training)

This implementation provides Knowledge Distillation (KD) training for ORION with EVA-ViT head pruning, reducing the number of attention heads from 16 to 8 (50% pruning). **Only the EVA-L backbone is trained**, while all other components (neck, detection heads, LLM components) are frozen.

## Overview

### Training Strategy
- **Teacher**: Full EVA-ViT with 16 attention heads (frozen)
- **Student**: Pruned EVA-ViT with 8 attention heads (trainable)
- **Frozen Components**: img_neck, pts_bbox_head, map_head, LLM components
- **Training Focus**: Only the student backbone learns from teacher backbone via KD

### Key Benefits
- **Faster Training**: Only ~25% of model parameters are trainable
- **Memory Efficient**: Reduced memory footprint during training
- **Stable Training**: Pre-trained components remain unchanged
- **Quick Convergence**: Backbone-only training converges in 2-3 epochs

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

### KD Parameters (Backbone-Focused)
```python
kd_config = dict(
    teacher_backbone_path='ckpt/orion/orion.pth',    # Full model (extracts backbone)
    distillation_alpha=0.7,                          # High KD weight (backbone focus)
    distillation_temperature=3.0,                    # Tight teacher following
    feature_distill_weight=0.5,                      # High feature distillation
    attention_distill_weight=0.3,                    # Attention transfer
    output_distill_weight=0.2,                       # Feature map matching
    warmup_epochs=1,                                 # Short warmup
    freeze_non_backbone=True,                        # Freeze non-backbone components
)
```

### Model Configuration (Backbone-Only Training)
```python
model = dict(
    type='OrionStudent',                             # Student model
    teacher_backbone_path=kd_config['teacher_backbone_path'],
    freeze_non_backbone=True,                        # Key: freeze everything else
    img_backbone=dict(
        type='EVAViTStudent',                        # Pruned backbone
        num_heads=8,                                 # Student heads
        teacher_num_heads=16,                        # Teacher heads
        # ... other params loaded from pre-trained
    ),
    # All other components loaded from pre-trained and frozen
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

### Backbone-Only Training
1. **Initialization**:
   - Load full pre-trained ORION model
   - Extract and freeze all components except student backbone
   - Load teacher EVA-L backbone (16 heads)
   - Initialize student EVA-L backbone (8 heads) from teacher

2. **Training Loop**:
   - Forward pass through student backbone → frozen components
   - Forward pass through teacher backbone (frozen)
   - Compute task loss (gradients only flow to student backbone)
   - Compute backbone distillation loss
   - Combined loss = 0.3 * task_loss + 0.7 * distillation_loss

### Loss Function (Backbone-Focused)
```python
total_loss = (1-α) * student_task_loss + α * backbone_distillation_loss

backbone_distillation_loss = λ₁ * feature_loss + λ₂ * attention_loss + λ₃ * feature_map_loss
```

### Training Duration
- **Total**: 3 epochs (much faster than full training)
- **Warmup**: 1 epoch with α=0.8
- **Fine-tuning**: 2 epochs with α=0.6

## Expected Results

### Performance Metrics
- **Model Size**: ~50% reduction in attention parameters
- **Inference Speed**: 20-30% faster
- **Memory Usage**: 25-40% reduction
- **Task Performance**: <5% degradation expected

### Training Logs (Backbone-Only)
Monitor these key metrics during training:
- `task_loss`: Original ORION task loss (from frozen components)
- `backbone_distill_loss`: Feature distillation between backbones
- `feature_distill_loss`: Intermediate feature matching
- `attention_distill_loss`: Attention pattern transfer
- `total_loss`: Combined task + distillation loss

Example log:
```
Iter [100/915] lr: 2.000e-04, eta: 1:15:23, time: 2.156, data_time: 0.089, memory: 8432,
task_loss: 0.5234, backbone_distill_loss: 0.1876, feature_distill_loss: 0.0934, 
attention_distill_loss: 0.0672, total_loss: 0.2882
```

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

## Repository

Original ORION repository: https://github.com/xiaomi-mlab/Orion  
This KD implementation: https://github.com/satabios/ORION

## Contact

For questions about the KD implementation, please open an issue or contact the maintainers.