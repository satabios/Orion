# ------------------------------------------------------------------------
# Knowledge Distillation Training Script for ORION
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.utils import get_logger
from mmcv.runner import build_optimizer, build_runner
from mmcv.datasets import build_dataset, build_dataloader


class KnowledgeDistillationTrainer:
    """Knowledge Distillation trainer for ORION student model."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger('KD_Trainer')
        
    def build_teacher_model(self):
        """Build and load teacher model."""
        from mmcv.models import build_model
        
        # Create teacher config (same as student but with full heads)
        teacher_cfg = self.cfg.model.copy()
        teacher_cfg.type = 'Orion'  # Use original Orion model
        teacher_cfg.img_backbone.type = 'EVAViT'  # Use original EVA-ViT
        teacher_cfg.img_backbone.num_heads = 16  # Full heads
        
        # Build teacher model
        teacher_model = build_model(
            teacher_cfg,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg')
        )
        
        # Load teacher weights
        teacher_checkpoint = torch.load(
            self.cfg.kd_config.teacher_model_path, 
            map_location='cpu'
        )
        
        if 'state_dict' in teacher_checkpoint:
            state_dict = teacher_checkpoint['state_dict']
        else:
            state_dict = teacher_checkpoint
            
        teacher_model.load_state_dict(state_dict, strict=False)
        
        # Freeze teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        
        self.logger.info("Teacher model loaded and frozen")
        return teacher_model
    
    def compute_head_importance_scores(self, teacher_model):
        """
        Compute importance scores for attention heads in teacher model.
        
        Returns:
            selected_heads_per_layer: Dict mapping layer index to selected head indices
        """
        from mmcv.models.backbones.eva_vit_student import compute_head_importance
        
        selected_heads_per_layer = {}
        teacher_backbone = teacher_model.img_backbone
        
        self.logger.info("Computing head importance scores...")
        
        for layer_idx, block in enumerate(teacher_backbone.blocks):
            if hasattr(block, 'attn'):
                attn = block.attn
                
                # Get QKV weights
                q_weight = attn.q_proj.weight.T  # [embed_dim, all_head_dim]
                k_weight = attn.k_proj.weight.T
                v_weight = attn.v_proj.weight.T
                
                # Compute importance scores
                importance_scores, head_indices = compute_head_importance(
                    q_weight, k_weight, v_weight, num_heads=16
                )
                
                # Select top 8 heads (50% pruning)
                selected_heads = head_indices[:8].tolist()
                selected_heads_per_layer[layer_idx] = selected_heads
                
                self.logger.info(
                    f"Layer {layer_idx}: Selected heads {selected_heads} "
                    f"with scores {importance_scores[head_indices[:8]].tolist()}"
                )
        
        return selected_heads_per_layer
    
    def initialize_student_from_teacher(self, student_model, teacher_model, selected_heads_per_layer):
        """Initialize student model weights from teacher model."""
        self.logger.info("Initializing student model from teacher...")
        
        # Store head selection in student model
        student_model.selected_heads_per_layer = selected_heads_per_layer
        
        # Transfer backbone weights
        if hasattr(student_model.img_backbone, 'transfer_teacher_weights'):
            student_model.img_backbone.transfer_teacher_weights(teacher_model.img_backbone)
        
        # Transfer other components
        self._transfer_component_weights(student_model.img_neck, teacher_model.img_neck, 'img_neck')
        self._transfer_component_weights(student_model.pts_bbox_head, teacher_model.pts_bbox_head, 'pts_bbox_head')
        
        if hasattr(student_model, 'map_head') and hasattr(teacher_model, 'map_head'):
            self._transfer_component_weights(student_model.map_head, teacher_model.map_head, 'map_head')
        
        self.logger.info("Student initialization completed")
    
    def _transfer_component_weights(self, student_component, teacher_component, component_name):
        """Transfer weights from teacher component to student component."""
        if student_component is not None and teacher_component is not None:
            try:
                student_component.load_state_dict(teacher_component.state_dict(), strict=False)
                self.logger.info(f"Transferred {component_name} weights")
            except Exception as e:
                self.logger.warning(f"Failed to transfer {component_name} weights: {e}")
    
    def train_with_kd(self, student_model, datasets):
        """Main training loop with knowledge distillation."""
        # Build teacher model
        teacher_model = self.build_teacher_model()
        
        # Compute head importance and initialize student
        selected_heads_per_layer = self.compute_head_importance_scores(teacher_model)
        self.initialize_student_from_teacher(student_model, teacher_model, selected_heads_per_layer)
        
        # Store teacher model in student for KD training
        student_model.teacher_model = teacher_model
        student_model.selected_heads_per_layer = selected_heads_per_layer
        
        # Build data loaders
        data_loaders = [
            build_dataloader(
                ds,
                self.cfg.data.samples_per_gpu,
                self.cfg.data.workers_per_gpu,
                len(self.cfg.gpu_ids),
                dist=True,
                seed=self.cfg.seed,
                shuffler_sampler=self.cfg.data.shuffler_sampler,
                nonshuffler_sampler=self.cfg.data.nonshuffler_sampler,
            ) for ds in datasets
        ]
        
        # Build optimizer
        optimizer = build_optimizer(student_model, self.cfg.optimizer)
        
        # Build runner
        runner = build_runner(
            self.cfg.runner,
            default_args=dict(
                model=student_model,
                optimizer=optimizer,
                work_dir=self.cfg.work_dir,
                logger=self.logger,
                meta={}
            )
        )
        
        # Register hooks
        runner.register_training_hooks(
            self.cfg.lr_config,
            self.cfg.optimizer_config,
            self.cfg.checkpoint_config,
            self.cfg.log_config,
            self.cfg.get('momentum_config', None)
        )
        
        # Custom KD hook
        if hasattr(self.cfg, 'custom_hooks'):
            for hook_cfg in self.cfg.custom_hooks:
                if hook_cfg.get('type') == 'TeacherStudentHook':
                    hook = TeacherStudentHook(**hook_cfg)
                    runner.register_hook(hook)
        
        # Load checkpoint if resuming
        if self.cfg.resume_from:
            runner.resume(self.cfg.resume_from)
        elif self.cfg.load_from:
            runner.load_checkpoint(self.cfg.load_from)
        
        # Start training
        runner.run(data_loaders, self.cfg.workflow)


class TeacherStudentHook:
    """Custom hook for teacher-student training management."""
    
    def __init__(self, teacher_model_path, priority='NORMAL'):
        self.teacher_model_path = teacher_model_path
        self.priority = priority
    
    def before_train_epoch(self, runner):
        """Ensure teacher model is in eval mode."""
        if hasattr(runner.model, 'teacher_model') and runner.model.teacher_model is not None:
            runner.model.teacher_model.eval()
    
    def before_train_iter(self, runner):
        """Inject current epoch into forward call for loss scheduling."""
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
            
        # Store current epoch for loss scheduling
        if hasattr(model, 'forward_train'):
            original_forward = model.forward_train
            
            def forward_with_epoch(*args, **kwargs):
                kwargs['current_epoch'] = runner.epoch
                return original_forward(*args, **kwargs)
            
            model.forward_train = forward_with_epoch


def custom_train_model_kd(model, datasets, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Custom training function for knowledge distillation."""
    logger = get_logger('KD_Training')
    
    # Create KD trainer
    kd_trainer = KnowledgeDistillationTrainer(cfg)
    
    # Start KD training
    kd_trainer.train_with_kd(model, datasets)
    
    logger.info("Knowledge Distillation training completed!")


# Utility functions for head pruning
def analyze_head_importance(model, dataloader, num_samples=100):
    """
    Analyze attention head importance using gradient-based method.
    
    Args:
        model: Teacher model
        dataloader: Training data loader
        num_samples: Number of samples to analyze
        
    Returns:
        head_importance: Dict with importance scores per layer
    """
    model.eval()
    head_importance = {}
    
    # Hook to capture gradients
    def capture_gradients(module, grad_input, grad_output, layer_idx):
        if layer_idx not in head_importance:
            head_importance[layer_idx] = []
        
        if hasattr(module, 'attn') and grad_output[0] is not None:
            # Compute head-wise gradient norms
            grad = grad_output[0]
            B, N, C = grad.shape
            num_heads = module.attn.num_heads
            head_dim = C // num_heads
            
            grad_heads = grad.view(B, N, num_heads, head_dim)
            head_grads = grad_heads.norm(dim=(0, 1, 3))  # [num_heads]
            head_importance[layer_idx].append(head_grads.detach().cpu())
    
    # Register hooks
    hooks = []
    for layer_idx, block in enumerate(model.img_backbone.blocks):
        hook = block.register_backward_hook(
            lambda module, grad_in, grad_out, idx=layer_idx: capture_gradients(
                module, grad_in, grad_out, idx
            )
        )
        hooks.append(hook)
    
    # Run forward passes with backprop
    with torch.enable_grad():
        for i, data_batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            model.zero_grad()
            losses = model.forward_train(**data_batch)
            loss = sum(losses.values())
            loss.backward()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Aggregate importance scores
    final_importance = {}
    for layer_idx, scores_list in head_importance.items():
        if scores_list:
            final_importance[layer_idx] = torch.stack(scores_list).mean(dim=0)
    
    return final_importance