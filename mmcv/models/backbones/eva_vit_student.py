# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from EVA-ViT for Knowledge Distillation with Head Pruning
# ------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

from ..builder import BACKBONES
from .eva_vit import (
    PatchEmbed, DropPath, SwiGLU, VisionRotaryEmbeddingFast,
    get_abs_pos, broadcat, rotate_half
)

try:
    from flash_attn.flash_attn_interface import FlashAttention
except:
    FlashAttention = None


def compute_head_importance(q_proj_weight, k_proj_weight, v_proj_weight, num_heads):
    """
    Compute importance scores for attention heads based on weight magnitudes.
    
    Args:
        q_proj_weight: Query projection weights [embed_dim, all_head_dim]
        k_proj_weight: Key projection weights [embed_dim, all_head_dim]  
        v_proj_weight: Value projection weights [embed_dim, all_head_dim]
        num_heads: Number of attention heads
        
    Returns:
        importance_scores: Tensor of shape [num_heads] with importance scores
        head_indices: Indices of heads sorted by importance (descending)
    """
    head_dim = q_proj_weight.size(1) // num_heads
    importance_scores = []
    
    for head_idx in range(num_heads):
        start_idx = head_idx * head_dim
        end_idx = (head_idx + 1) * head_dim
        
        # Extract weights for this head
        q_head = q_proj_weight[:, start_idx:end_idx]
        k_head = k_proj_weight[:, start_idx:end_idx]  
        v_head = v_proj_weight[:, start_idx:end_idx]
        
        # Compute L2 norm of combined QKV weights for this head
        head_importance = (q_head.norm(p=2) + k_head.norm(p=2) + v_head.norm(p=2)) / 3.0
        importance_scores.append(head_importance.item())
    
    importance_scores = torch.tensor(importance_scores)
    head_indices = torch.argsort(importance_scores, descending=True)
    
    return importance_scores, head_indices


class PrunedAttention(nn.Module):
    """Pruned Attention module with reduced number of heads."""
    
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=True, 
        qk_scale=None, 
        attn_head_dim=None, 
        norm_layer=nn.LayerNorm,
        rope=None,
        flash_attn=True,
        subln=False,
        teacher_num_heads=16,
        selected_heads=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.teacher_num_heads = teacher_num_heads
        self.selected_heads = selected_heads
        
        head_dim = dim // teacher_num_heads  # Use teacher's head dim for compatibility
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rope = rope
        self.flash_attn = flash_attn
        self.proj = nn.Linear(all_head_dim, dim)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()

        if self.flash_attn and FlashAttention is not None:
            factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
            self.inner_attn = FlashAttention(attention_dropout=0.0, **factory_kwargs)

    def transfer_teacher_weights(self, teacher_attention):
        """Transfer selected head weights from teacher to student."""
        if self.selected_heads is None:
            # Compute head importance and select top heads
            with torch.no_grad():
                importance_scores, head_indices = compute_head_importance(
                    teacher_attention.q_proj.weight.T,
                    teacher_attention.k_proj.weight.T, 
                    teacher_attention.v_proj.weight.T,
                    self.teacher_num_heads
                )
                self.selected_heads = head_indices[:self.num_heads]
        
        teacher_head_dim = teacher_attention.q_proj.weight.size(0) // self.teacher_num_heads
        student_head_dim = self.q_proj.weight.size(0) // self.num_heads
        
        # Transfer Q, K, V projection weights for selected heads
        with torch.no_grad():
            for i, head_idx in enumerate(self.selected_heads):
                teacher_start = head_idx * teacher_head_dim
                teacher_end = (head_idx + 1) * teacher_head_dim
                student_start = i * student_head_dim
                student_end = (i + 1) * student_head_dim
                
                # Transfer weights
                self.q_proj.weight[student_start:student_end].copy_(
                    teacher_attention.q_proj.weight[teacher_start:teacher_end]
                )
                self.k_proj.weight[student_start:student_end].copy_(
                    teacher_attention.k_proj.weight[teacher_start:teacher_end]
                )
                self.v_proj.weight[student_start:student_end].copy_(
                    teacher_attention.v_proj.weight[teacher_start:teacher_end]
                )
                
                # Transfer biases if they exist
                if self.q_bias is not None and teacher_attention.q_bias is not None:
                    self.q_bias[student_start:student_end].copy_(
                        teacher_attention.q_bias[teacher_start:teacher_end]
                    )
                if self.v_bias is not None and teacher_attention.v_bias is not None:
                    self.v_bias[student_start:student_end].copy_(
                        teacher_attention.v_bias[teacher_start:teacher_end]
                    )
        
        # Transfer output projection (full weight since output dim is same)
        self.proj.weight.copy_(teacher_attention.proj.weight)
        if self.proj.bias is not None and teacher_attention.proj.bias is not None:
            self.proj.bias.copy_(teacher_attention.proj.bias)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        N = H * W

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 

        ## rope
        q = self.rope(q).type_as(v)
        k = self.rope(k).type_as(v)

        if self.flash_attn and FlashAttention is not None:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            kv = torch.stack([k, v], dim=2)
            x, attn_weights = self.inner_attn(q, kv, key_padding_mask=None, causal=False)
            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1).type_as(x)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)

        x = self.proj(x)
        x = x.view(B, H, W, C)

        return x


class PrunedBlock(nn.Module):
    """Transformer block with pruned attention heads."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        window_size=0,
        use_residual_block=False,
        rope=None,
        flash_attn=True,
        subln=False,
        with_cp=True,
        teacher_num_heads=16,
        selected_heads=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrunedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rope=rope,
            flash_attn=flash_attn,
            subln=subln,
            teacher_num_heads=teacher_num_heads,
            selected_heads=selected_heads
        )

        self.with_cp = with_cp
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            subln=True,
            norm_layer=norm_layer,
        )

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            from .eva_vit import ResBottleneckBlock
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
            )

    def _forward(self, x):
        if self.use_residual_block:
            x = x + self.drop_path(self.attn(self.norm1(x))) + self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as cp
            x = cp.checkpoint(self._forward, x)
        else:
            x = self._forward(x)
        return x

    def transfer_teacher_weights(self, teacher_block):
        """Transfer weights from teacher block to student block."""
        # Transfer attention weights
        self.attn.transfer_teacher_weights(teacher_block.attn)
        
        # Transfer other components (norm layers, MLP)
        self.norm1.weight.copy_(teacher_block.norm1.weight)
        self.norm1.bias.copy_(teacher_block.norm1.bias)
        self.norm2.weight.copy_(teacher_block.norm2.weight) 
        self.norm2.bias.copy_(teacher_block.norm2.bias)
        
        # Transfer MLP weights
        if hasattr(teacher_block.mlp, 'w12'):
            self.mlp.w12.weight.copy_(teacher_block.mlp.w12.weight)
            if self.mlp.w12.bias is not None:
                self.mlp.w12.bias.copy_(teacher_block.mlp.w12.bias)
        if hasattr(teacher_block.mlp, 'w3'):
            self.mlp.w3.weight.copy_(teacher_block.mlp.w3.weight)
            if self.mlp.w3.bias is not None:
                self.mlp.w3.bias.copy_(teacher_block.mlp.w3.bias)


@BACKBONES.register_module()
class EVAViTStudent(nn.Module):
    """
    EVA-ViT Student model with pruned attention heads for Knowledge Distillation.
    
    This model reduces the number of attention heads by 50% (from 16 to 8 heads)
    while maintaining the same overall architecture and feature dimensions.
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=8,  # Reduced from 16 to 8
        teacher_num_heads=16,  # Original number of heads
        mlp_ratio=4*2/3,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        window_size=0,
        global_window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        subln=False,
        flash_attn=True,
        with_cp=True,
        frozen=False,
        selected_heads_per_layer=None,  # Pre-computed head selections
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.teacher_num_heads = teacher_num_heads
        self.num_heads = num_heads
        self.selected_heads_per_layer = selected_heads_per_layer

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        half_head_dim = embed_dim // teacher_num_heads // 2  # Use teacher's head dim
        hw_seq_len = img_size // patch_size

        self.rope_win = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=window_size if intp_freq else None,
        )
        self.rope_glb = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=hw_seq_len if intp_freq else None,
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            selected_heads = None
            if selected_heads_per_layer is not None:
                selected_heads = selected_heads_per_layer.get(i, None)
                
            block = PrunedBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=window_size if i in window_block_indexes else global_window_size,
                use_residual_block=i in residual_block_indexes,
                rope=self.rope_win if i in window_block_indexes else self.rope_glb,
                flash_attn=flash_attn,
                subln=subln,
                with_cp=with_cp,
                teacher_num_heads=teacher_num_heads,
                selected_heads=selected_heads,
            )

            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)

        self.frozen = frozen

    def _freeze_stages(self):
        if self.frozen:
            self.eval()
            for m in self.parameters():
                m.requires_grad = False

    def transfer_teacher_weights(self, teacher_model):
        """Transfer weights from teacher EVAViT to student model."""
        print("Transferring teacher weights to student model...")
        
        # Transfer patch embedding
        self.patch_embed.proj.weight.copy_(teacher_model.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            self.patch_embed.proj.bias.copy_(teacher_model.patch_embed.proj.bias)
        
        # Transfer positional embeddings
        if self.pos_embed is not None and teacher_model.pos_embed is not None:
            self.pos_embed.copy_(teacher_model.pos_embed)
        
        # Transfer block weights
        for i, (student_block, teacher_block) in enumerate(zip(self.blocks, teacher_model.blocks)):
            print(f"Transferring block {i}/{len(self.blocks)}")
            student_block.transfer_teacher_weights(teacher_block)
        
        print("Teacher weight transfer completed!")

    def forward(self, x):
        dtype = self.patch_embed.proj.weight.data.dtype
        x = x.to(dtype)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            ).to(x.dtype)

        for blk in self.blocks:
            x = blk(x)   # b, h, w, c
        x = x.permute(0, 3, 1, 2) # b, c, h, w 
        
        return [x]