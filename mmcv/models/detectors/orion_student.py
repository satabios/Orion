# ------------------------------------------------------------------------
# ORION Student Model with Knowledge Distillation
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.utils import auto_fp16
from mmcv.models import DETECTORS
from mmcv.models.detectors.orion import Orion
from mmcv.models.builder import build_loss
from mmcv.utils.misc import load_model


@DETECTORS.register_module()
class OrionStudent(Orion):
    """
    ORION Student model for Knowledge Distillation.
    
    This model uses a pruned EVA-ViT backbone with 50% fewer attention heads
    and is trained using knowledge distillation from the full teacher model.
    """
    
    def __init__(self,
                 teacher_model_path=None,
                 distillation_loss=None,
                 distillation_alpha=0.4,
                 selected_heads_per_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.teacher_model_path = teacher_model_path
        self.teacher_model = None
        self.distillation_alpha = distillation_alpha
        self.selected_heads_per_layer = selected_heads_per_layer or {}
        
        # Build distillation loss
        if distillation_loss is not None:
            self.distillation_loss = build_loss(distillation_loss)
        else:
            self.distillation_loss = None
            
    def load_teacher_model(self):
        """Load and freeze teacher model."""
        if self.teacher_model is None and self.teacher_model_path is not None:
            print(f"Loading teacher model from {self.teacher_model_path}")
            
            # Create teacher model with same config but full heads
            teacher_config = self.cfg.copy()
            teacher_config.model.img_backbone.num_heads = 16  # Full heads for teacher
            
            # Load teacher model
            from mmcv.models import build_model
            self.teacher_model = build_model(
                teacher_config.model,
                train_cfg=teacher_config.get('train_cfg'),
                test_cfg=teacher_config.get('test_cfg')
            )
            
            # Load teacher weights
            checkpoint = torch.load(self.teacher_model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            self.teacher_model.load_state_dict(state_dict, strict=False)
            
            # Freeze teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            
            print("Teacher model loaded and frozen")
            
    def init_student_from_teacher(self):
        """Initialize student model weights from teacher model."""
        if self.teacher_model is None:
            self.load_teacher_model()
            
        if hasattr(self.img_backbone, 'transfer_teacher_weights'):
            print("Transferring teacher weights to student backbone...")
            self.img_backbone.transfer_teacher_weights(self.teacher_model.img_backbone)
        
        # Transfer other components (img_neck, bbox_head, etc.)
        if self.img_neck is not None and self.teacher_model.img_neck is not None:
            print("Transferring img_neck weights...")
            self.img_neck.load_state_dict(self.teacher_model.img_neck.state_dict(), strict=False)
            
        if self.pts_bbox_head is not None and self.teacher_model.pts_bbox_head is not None:
            print("Transferring bbox_head weights...")
            self.pts_bbox_head.load_state_dict(self.teacher_model.pts_bbox_head.state_dict(), strict=False)
            
        if hasattr(self, 'map_head') and hasattr(self.teacher_model, 'map_head'):
            if self.map_head is not None and self.teacher_model.map_head is not None:
                print("Transferring map_head weights...")
                self.map_head.load_state_dict(self.teacher_model.map_head.state_dict(), strict=False)
        
        print("Student model initialization from teacher completed!")
        
    def extract_student_features(self, img_metas, gt_bboxes_3d, gt_labels_3d, 
                                gt_attr_labels, map_gt_bboxes_3d, map_gt_labels_3d,
                                input_ids, vlm_labels, ego_fut_trajs, **data):
        """Extract features from student model."""
        # Forward through student model
        img_feats = self.extract_img_feat(data['img'], img_metas, **data)
        
        # Get bbox head outputs
        student_outputs = self.pts_bbox_head(
            img_feats, img_metas, gt_bboxes_3d, gt_labels_3d, gt_attr_labels,
            map_gt_bboxes_3d, map_gt_labels_3d, input_ids, vlm_labels, 
            ego_fut_trajs, **data
        )
        
        return {
            'img_feats': img_feats,
            'bbox_outputs': student_outputs,
            'intermediate_features': getattr(self.img_backbone, 'intermediate_features', [])
        }
    
    def extract_teacher_features(self, img_metas, gt_bboxes_3d, gt_labels_3d,
                                gt_attr_labels, map_gt_bboxes_3d, map_gt_labels_3d, 
                                input_ids, vlm_labels, ego_fut_trajs, **data):
        """Extract features from teacher model."""
        if self.teacher_model is None:
            self.load_teacher_model()
            
        self.teacher_model.eval()
        with torch.no_grad():
            # Forward through teacher model
            teacher_img_feats = self.teacher_model.extract_img_feat(data['img'], img_metas, **data)
            
            # Get teacher bbox head outputs
            teacher_outputs = self.teacher_model.pts_bbox_head(
                teacher_img_feats, img_metas, gt_bboxes_3d, gt_labels_3d, gt_attr_labels,
                map_gt_bboxes_3d, map_gt_labels_3d, input_ids, vlm_labels,
                ego_fut_trajs, **data
            )
        
        return {
            'img_feats': teacher_img_feats,
            'bbox_outputs': teacher_outputs,
            'intermediate_features': getattr(self.teacher_model.img_backbone, 'intermediate_features', [])
        }

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_attr_labels=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      input_ids=None,
                      vlm_labels=None,
                      ego_fut_trajs=None,
                      current_epoch=0,
                      **data):
        """
        Forward training with knowledge distillation.
        
        Args:
            current_epoch: Current training epoch for loss scheduling
        """
        # Extract student features and outputs
        student_results = self.extract_student_features(
            img_metas, gt_bboxes_3d, gt_labels_3d, gt_attr_labels,
            map_gt_bboxes_3d, map_gt_labels_3d, input_ids, vlm_labels,
            ego_fut_trajs, **data
        )
        
        # Compute student task loss (original ORION loss)
        student_task_losses = student_results['bbox_outputs']
        student_task_loss = sum(student_task_losses.values())
        
        # If distillation is enabled, compute distillation loss
        if self.distillation_loss is not None and self.training:
            # Extract teacher features
            teacher_results = self.extract_teacher_features(
                img_metas, gt_bboxes_3d, gt_labels_3d, gt_attr_labels,
                map_gt_bboxes_3d, map_gt_labels_3d, input_ids, vlm_labels,
                ego_fut_trajs, **data
            )
            
            # Prepare distillation inputs
            distill_kwargs = {
                'student_features': student_results.get('intermediate_features', []),
                'teacher_features': teacher_results.get('intermediate_features', []),
                'selected_heads_per_layer': self.selected_heads_per_layer,
                'current_epoch': current_epoch,
                'student_task_loss': student_task_loss,
            }
            
            # Add output distillation if both models have compatible outputs
            if 'cls_scores' in student_results['bbox_outputs'] and 'cls_scores' in teacher_results['bbox_outputs']:
                distill_kwargs['student_logits'] = student_results['bbox_outputs']['cls_scores']
                distill_kwargs['teacher_logits'] = teacher_results['bbox_outputs']['cls_scores']
            
            # Compute distillation loss
            if hasattr(self.distillation_loss, 'forward'):
                total_loss, distill_loss_dict = self.distillation_loss(**distill_kwargs)
            else:
                # Simple alpha-weighted combination
                distill_loss = student_task_loss  # Placeholder for actual distillation
                total_loss = (1 - self.distillation_alpha) * student_task_loss + self.distillation_alpha * distill_loss
                distill_loss_dict = {'distill_loss': distill_loss}
            
            # Combine losses
            all_losses = {}
            all_losses.update(student_task_losses)
            all_losses.update(distill_loss_dict)
            all_losses['total_loss'] = total_loss
            
            return all_losses
        else:
            # Return original student losses without distillation
            return student_task_losses

    def train(self, mode=True):
        """Set training mode and handle teacher model."""
        super().train(mode)
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            self.teacher_model.eval()  # Keep teacher in eval mode
        return self

    def cuda(self, device=None):
        """Move to CUDA and handle teacher model."""
        result = super().cuda(device)
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            self.teacher_model = self.teacher_model.cuda(device)
        return result

    def to(self, device):
        """Move to device and handle teacher model."""
        result = super().to(device)
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(device)
        return result