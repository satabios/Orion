# ------------------------------------------------------------------------
# ORION Student Model with Knowledge Distillation
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from mmcv.runner.checkpoint import load_checkpoint
from mmcv.models import DETECTORS
from mmcv.models.detectors.orion import Orion
from mmcv.models.builder import build_loss
from mmcv.utils.misc import load_model


@DETECTORS.register_module()
class OrionStudent(Orion):
    """
    ORION Student model for Knowledge Distillation.
    
    This model uses a pruned EVA-ViT backbone with 50% fewer attention heads.
    Only the EVA-L backbone is trained via KD, all other components are frozen.
    """
    
    def __init__(self,
                 teacher_backbone_path=None,  # Path to full teacher model
                 distillation_loss=None,
                 distillation_alpha=0.4,
                 selected_heads_per_layer=None,
                 freeze_non_backbone=True,  # Freeze everything except backbone
                 **kwargs): 
        super().__init__(**kwargs)
        
        self.teacher_backbone_path = teacher_backbone_path
        self.teacher_backbone = None  # Only load teacher backbone, not full model
        self.distillation_alpha = distillation_alpha
        self.selected_heads_per_layer = selected_heads_per_layer or {}
        self.freeze_non_backbone = freeze_non_backbone
        
        # Build distillation loss
        if distillation_loss is not None:
            self.distillation_loss = build_loss(distillation_loss)
        else:
            self.distillation_loss = None
            
        # This will be called by the runner after loading the student checkpoint
        self.init_student_from_teacher()

    def _freeze_by_prefix(self, prefix):
        """Helper to freeze parameters by prefix."""
        count = 0
        for name, param in self.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = False
                count += 1
        if count > 0:
            print(f"  - Froze {count} parameters under '{prefix}'")

    def freeze_non_backbone_components(self, model):
        """Freeze all components except the student backbone."""
        if not self.freeze_non_backbone or not hasattr(model, 'img_backbone'):
            return
            
        print("Freezing non-backbone components...")
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the student backbone
        for param in model.img_backbone.parameters():
            param.requires_grad = True

        print("Non-backbone components frozen. Only student backbone will be trained.")
            
    def init_student_from_teacher(self):
        """
        Initializes the student model.
        1. Creates a teacher model from the original Orion config.
        2. Loads full pre-trained weights into the teacher.
        3. Transfers weights from the teacher backbone to the student backbone.
        4. Freezes non-backbone components of the student model.
        """
        if self.teacher_backbone_path is None:
            print("No teacher model path provided. Skipping KD initialization.")
            return

        print("--- Initializing Student Model for Knowledge Distillation ---")
        # 1. Build teacher model (original Orion)
        from mmcv.models.backbones import EVAViT
        teacher_model = Orion(**self._get_teacher_kwargs())
        
        # 2. Load full pre-trained weights
        print(f"Loading full teacher model from: {self.teacher_backbone_path}")
        load_checkpoint(teacher_model, self.teacher_backbone_path, map_location='cpu', strict=False)
        
        # 3. Transfer backbone weights
        print("Transferring teacher backbone weights to student backbone...")
        if hasattr(self.img_backbone, 'transfer_teacher_weights'):
            self.img_backbone.transfer_teacher_weights(teacher_model.img_backbone)
        
        # 4. Freeze non-backbone components in the student model
        self.freeze_non_backbone_components(self)

        # 5. Store the teacher's backbone, freeze it, and put it in eval mode
        self.teacher_backbone = teacher_model.img_backbone
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        self.teacher_backbone.eval()
        print("Teacher backbone is ready and frozen.")
        print("--- Student Initialization Complete ---")

    def _get_teacher_kwargs(self):
        """Get kwargs for building the teacher model, ensuring the backbone is the original EVAViT."""
        kwargs = self.student_cfg
        # Override student-specific backbone with the original one
        kwargs['img_backbone']['type'] = 'EVAViT'
        kwargs['img_backbone']['num_heads'] = 16
        # Remove student-specific keys
        kwargs['img_backbone'].pop('teacher_num_heads', None)
        return kwargs

    @torch.no_grad()
    def extract_teacher_features(self, img):
        """Extract features from teacher EVA-L backbone only."""
        if self.teacher_backbone is None:
            raise RuntimeError("Teacher backbone not loaded. Call init_student_from_teacher first.")
        self.teacher_backbone.eval()
        teacher_img_feats = self.teacher_backbone(img)
        return {
            'img_feats': teacher_img_feats,
             'backbone_features': getattr(self.teacher_backbone, 'intermediate_features', [])
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
        Forward training with backbone-only knowledge distillation.
        
        Only the EVA-L backbone is trained via KD, everything else is frozen.
        """
        # 1. Forward student model to get task losses
        # Gradients will only flow through the student backbone as other parts are frozen.
        task_losses = super().forward_train(img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d,
                                            gt_labels_3d=gt_labels_3d, gt_attr_labels=gt_attr_labels,
                                            map_gt_bboxes_3d=map_gt_bboxes_3d, map_gt_labels_3d=map_gt_labels_3d,
                                            input_ids=input_ids, vlm_labels=vlm_labels,
                                            ego_fut_trajs=ego_fut_trajs, **data)

        student_task_loss = sum(task_losses.values())

        # 2. Compute distillation loss
        if self.distillation_loss is not None and self.training:
            # Extract features from student and teacher backbones
            student_backbone_feats = self.img_backbone.intermediate_features
            student_output_feats = self.img_backbone.last_features

            teacher_results = self.extract_teacher_features(data['img'])
            teacher_backbone_feats = teacher_results.get('backbone_features', [])
            teacher_output_feats = teacher_results.get('img_feats', [None])[0]

            distill_kwargs = {
                'student_features': student_backbone_feats,
                'teacher_features': teacher_backbone_feats,
                'student_logits': student_output_feats,
                'teacher_logits': teacher_output_feats,
                'selected_heads_per_layer': self.selected_heads_per_layer,
            }

            distill_loss, distill_loss_dict = self.distillation_loss(**distill_kwargs)

            # 3. Combine losses
            alpha = self.distillation_alpha
            total_loss = (1 - alpha) * student_task_loss + alpha * distill_loss

            # Log all losses
            final_losses = {}
            final_losses.update(task_losses)
            final_losses.update(distill_loss_dict)
            final_losses['total_loss'] = total_loss # Overwrite orion's total_loss
            final_losses['distill_loss'] = distill_loss # For logging

            return final_losses
        else:
            # Return original task losses without distillation
            return task_losses

    def train(self, mode=True):
        """Set training mode and handle teacher backbone."""
        super().train(mode)
        if hasattr(self, 'teacher_backbone') and self.teacher_backbone is not None:
            self.teacher_backbone.eval()  # Keep teacher in eval mode
        return self

    def cuda(self, device=None):
        """Move to CUDA and handle teacher backbone."""
        super().cuda(device)
        if hasattr(self, 'teacher_backbone') and self.teacher_backbone is not None:
            self.teacher_backbone = self.teacher_backbone.cuda(device)
        return self

    def to(self, device):
        """Move to device and handle teacher backbone."""
        super().to(device)
        if hasattr(self, 'teacher_backbone') and self.teacher_backbone is not None:
            self.teacher_backbone = self.teacher_backbone.to(device)
        return self