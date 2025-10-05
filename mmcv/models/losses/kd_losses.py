# ------------------------------------------------------------------------
# Knowledge Distillation Loss Functions for ORION
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class FeatureDistillationLoss(nn.Module):
    """Feature-level distillation loss to match intermediate feature representations."""
    
    def __init__(self, loss_weight=1.0, temperature=4.0, distance_type='mse'):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.distance_type = distance_type
        
    def forward(self, student_features, teacher_features, **kwargs):
        """
        Args:
            student_features: List of feature tensors from student model
            teacher_features: List of feature tensors from teacher model
        """
        if len(student_features) != len(teacher_features):
            raise ValueError("Student and teacher must have same number of feature layers")
        
        loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            if s_feat.shape != t_feat.shape:
                # If shapes differ, align them
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
            
            if self.distance_type == 'mse':
                loss += F.mse_loss(s_feat, t_feat.detach())
            elif self.distance_type == 'cosine':
                s_feat_flat = s_feat.view(s_feat.size(0), -1)
                t_feat_flat = t_feat.view(t_feat.size(0), -1)
                loss += 1 - F.cosine_similarity(s_feat_flat, t_feat_flat.detach()).mean()
            else:
                raise ValueError(f"Unsupported distance type: {self.distance_type}")
        
        return self.loss_weight * loss / len(student_features)


@LOSSES.register_module()
class AttentionDistillationLoss(nn.Module):
    """Attention distillation loss to transfer attention patterns from teacher to student."""
    
    def __init__(self, loss_weight=1.0, temperature=4.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        
    def forward(self, student_attentions, teacher_attentions, selected_heads_per_layer, **kwargs):
        """
        Args:
            student_attentions: List of attention tensors from student [B, num_heads, N, N]
            teacher_attentions: List of attention tensors from teacher [B, teacher_num_heads, N, N]
            selected_heads_per_layer: Dict mapping layer idx to selected head indices
        """
        if len(student_attentions) != len(teacher_attentions):
            raise ValueError("Student and teacher must have same number of attention layers")
        
        loss = 0.0
        for layer_idx, (s_attn, t_attn) in enumerate(zip(student_attentions, teacher_attentions)):
            if layer_idx in selected_heads_per_layer:
                selected_heads = selected_heads_per_layer[layer_idx]
                # Select corresponding teacher attention heads
                t_attn_selected = t_attn[:, selected_heads, :, :]
                
                # Apply temperature scaling and compute KL divergence
                s_attn_soft = F.log_softmax(s_attn / self.temperature, dim=-1)
                t_attn_soft = F.softmax(t_attn_selected.detach() / self.temperature, dim=-1)
                
                loss += F.kl_div(s_attn_soft, t_attn_soft, reduction='batchmean')
        
        return self.loss_weight * loss * (self.temperature ** 2) / len(student_attentions)


@LOSSES.register_module()
class OutputDistillationLoss(nn.Module):
    """Output distillation loss for soft target matching."""
    
    def __init__(self, loss_weight=1.0, temperature=4.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits, **kwargs):
        """
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits.detach() / self.temperature, dim=-1)
        
        # Compute KL divergence
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        return self.loss_weight * loss * (self.temperature ** 2)


@LOSSES.register_module()
class VisionLanguageDistillationLoss(nn.Module):
    """Vision-Language distillation loss for multimodal embeddings."""
    
    def __init__(self, loss_weight=1.0, temperature=4.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        
    def forward(self, student_vl_features, teacher_vl_features, **kwargs):
        """
        Args:
            student_vl_features: Vision-language features from student
            teacher_vl_features: Vision-language features from teacher
        """
        # Normalize features
        student_norm = F.normalize(student_vl_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_vl_features.detach(), p=2, dim=-1)
        
        # Compute cosine similarity loss
        loss = 1 - F.cosine_similarity(student_norm, teacher_norm).mean()
        
        return self.loss_weight * loss


@LOSSES.register_module()
class CombinedKDLoss(nn.Module):
    """Combined Knowledge Distillation Loss."""
    
    def __init__(self, 
                 feature_distill_weight=0.3,
                 attention_distill_weight=0.2, 
                 output_distill_weight=0.3,
                 vl_distill_weight=0.2,
                 temperature=4.0):
        super().__init__()
        
        self.feature_loss = FeatureDistillationLoss(
            loss_weight=feature_distill_weight, 
            temperature=temperature
        )
        self.attention_loss = AttentionDistillationLoss(
            loss_weight=attention_distill_weight,
            temperature=temperature
        )
        self.output_loss = OutputDistillationLoss(
            loss_weight=output_distill_weight,
            temperature=temperature
        )
        self.vl_loss = VisionLanguageDistillationLoss(
            loss_weight=vl_distill_weight,
            temperature=temperature
        )
        
    def forward(self, 
                student_features=None,
                teacher_features=None,
                student_attentions=None,
                teacher_attentions=None,
                selected_heads_per_layer=None,
                student_logits=None,
                teacher_logits=None,
                student_vl_features=None,
                teacher_vl_features=None,
                **kwargs):
        """
        Compute combined distillation loss.
        
        Args:
            student_features: List of student intermediate features
            teacher_features: List of teacher intermediate features  
            student_attentions: List of student attention maps
            teacher_attentions: List of teacher attention maps
            selected_heads_per_layer: Head selection mapping
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            student_vl_features: Student vision-language features
            teacher_vl_features: Teacher vision-language features
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Feature distillation
        if student_features is not None and teacher_features is not None:
            feat_loss = self.feature_loss(student_features, teacher_features)
            total_loss += feat_loss
            loss_dict['feature_distill_loss'] = feat_loss
        
        # Attention distillation
        if (student_attentions is not None and teacher_attentions is not None 
            and selected_heads_per_layer is not None):
            attn_loss = self.attention_loss(
                student_attentions, teacher_attentions, selected_heads_per_layer
            )
            total_loss += attn_loss
            loss_dict['attention_distill_loss'] = attn_loss
        
        # Output distillation
        if student_logits is not None and teacher_logits is not None:
            out_loss = self.output_loss(student_logits, teacher_logits)
            total_loss += out_loss
            loss_dict['output_distill_loss'] = out_loss
        
        # Vision-language distillation
        if student_vl_features is not None and teacher_vl_features is not None:
            vl_loss = self.vl_loss(student_vl_features, teacher_vl_features)
            total_loss += vl_loss
            loss_dict['vl_distill_loss'] = vl_loss
        
        loss_dict['total_distill_loss'] = total_loss
        return total_loss, loss_dict


@LOSSES.register_module()
class KDScheduledLoss(nn.Module):
    """Knowledge Distillation loss with scheduling."""
    
    def __init__(self, 
                 base_kd_loss,
                 warmup_epochs=2,
                 warmup_alpha=0.6,
                 final_alpha=0.3):
        super().__init__()
        self.kd_loss = base_kd_loss
        self.warmup_epochs = warmup_epochs
        self.warmup_alpha = warmup_alpha
        self.final_alpha = final_alpha
        
    def forward(self, current_epoch, student_task_loss, **distill_kwargs):
        """
        Args:
            current_epoch: Current training epoch
            student_task_loss: Task-specific loss from student model
            **distill_kwargs: Arguments for distillation loss
        """
        # Compute distillation loss
        distill_loss, loss_dict = self.kd_loss(**distill_kwargs)
        
        # Schedule alpha based on epoch
        if current_epoch < self.warmup_epochs:
            alpha = self.warmup_alpha
        else:
            alpha = self.final_alpha
        
        # Combined loss
        total_loss = (1 - alpha) * student_task_loss + alpha * distill_loss
        
        loss_dict.update({
            'student_task_loss': student_task_loss,
            'alpha': alpha,
            'total_kd_loss': total_loss
        })
        
        return total_loss, loss_dict