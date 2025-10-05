# ------------------------------------------------------------------------
# ORION Student Model with Knowledge Distillation
# ------------------------------------------------------------------------

import torch
import copy
import torch.nn.functional as F
from mmcv.utils import load_checkpoint
from mmcv.models import DETECTORS
from mmcv.models.detectors.orion import Orion
from mmcv.models.builder import build_loss



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
        # Store a copy of the student's configuration to build the teacher model later
        self.student_cfg = copy.deepcopy(kwargs)
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
        Initializes the student model by loading teacher backbone from checkpoint.
        1. Loads the teacher checkpoint directly.
        2. Extracts the teacher backbone from the checkpoint.
        3. Transfers weights from teacher backbone to student backbone.
        4. Freezes non-backbone components of the student model.
        """
        if self.teacher_backbone_path is None:
            print("No teacher model path provided. Skipping KD initialization.")
            return

        print("--- Initializing Student Model for Knowledge Distillation ---")
        
        try:
                        # 1. Load teacher checkpoint directly
            print(f"Loading teacher checkpoint from: {self.teacher_backbone_path}")
            checkpoint = torch.load(self.teacher_backbone_path, map_location='cpu')
            
            # Extract state dict
            if 'state_dict' in checkpoint:
                teacher_state_dict = checkpoint['state_dict']
            else:
                teacher_state_dict = checkpoint
            
            # 2. Extract teacher backbone state dict
            teacher_backbone_state = {}
            backbone_prefix = 'img_backbone.'
            for key, value in teacher_state_dict.items():
                if key.startswith(backbone_prefix):
                    # Remove the prefix to get the backbone parameter name
                    backbone_key = key[len(backbone_prefix):]
                    teacher_backbone_state[backbone_key] = value
            
            print(f"Extracted {len(teacher_backbone_state)} backbone parameters from teacher")
            
            # 3. Load teacher backbone weights into student backbone
            if teacher_backbone_state:
                print("Loading teacher backbone weights into student backbone...")
                missing_keys, unexpected_keys = self.img_backbone.load_state_dict(
                    teacher_backbone_state, strict=False
                )
                print(f"Loaded teacher weights: {len(teacher_backbone_state) - len(missing_keys)} matched")
                if missing_keys:
                    print(f"Missing keys (expected for pruned model): {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            print(f"Warning: Could not load teacher model: {e}")
            print("Continuing with random initialization...")
        
        # 4. Freeze non-backbone components regardless of teacher loading success
        if self.freeze_non_backbone:
            self.freeze_non_backbone_components()

        print("--- Student Initialization Complete ---")

    @torch.no_grad()
    def extract_teacher_features(self, img):
        """Extract features from teacher - disabled for checkpoint-based loading."""
        # For backbone-only KD, we don't need separate teacher inference
        # The teacher weights are transferred during initialization
        return None

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