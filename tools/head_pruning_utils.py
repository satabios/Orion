# ------------------------------------------------------------------------
# Head Pruning Utilities for ORION Knowledge Distillation
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from collections import defaultdict


def compute_head_importance_magnitude(model, save_path=None):
    """
    Compute attention head importance based on weight magnitudes.
    
    Args:
        model: Teacher model with full attention heads
        save_path: Path to save importance scores
        
    Returns:
        importance_dict: Dictionary with importance scores per layer
    """
    importance_dict = {}
    
    print("Computing head importance based on weight magnitudes...")
    
    for layer_idx, block in enumerate(model.img_backbone.blocks):
        if hasattr(block, 'attn'):
            attn = block.attn
            
            # Get attention projection weights
            q_weight = attn.q_proj.weight  # [all_head_dim, embed_dim]
            k_weight = attn.k_proj.weight  # [all_head_dim, embed_dim]
            v_weight = attn.v_proj.weight  # [all_head_dim, embed_dim]
            
            num_heads = attn.num_heads
            head_dim = q_weight.size(0) // num_heads
            
            head_scores = []
            for head_idx in range(num_heads):
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                
                # Extract weights for this head
                q_head = q_weight[start_idx:end_idx, :]
                k_head = k_weight[start_idx:end_idx, :]
                v_head = v_weight[start_idx:end_idx, :]
                
                # Compute L2 norm of combined weights
                q_norm = torch.norm(q_head, p=2)
                k_norm = torch.norm(k_head, p=2)
                v_norm = torch.norm(v_head, p=2)
                
                # Combined importance score
                head_importance = (q_norm + k_norm + v_norm) / 3.0
                head_scores.append(head_importance.item())
            
            # Sort heads by importance
            head_scores = torch.tensor(head_scores)
            sorted_indices = torch.argsort(head_scores, descending=True)
            
            importance_dict[layer_idx] = {
                'scores': head_scores.tolist(),
                'sorted_indices': sorted_indices.tolist(),
                'selected_heads_50': sorted_indices[:8].tolist(),  # Top 50%
                'selected_heads_25': sorted_indices[:4].tolist(),  # Top 25%
                'selected_heads_75': sorted_indices[:12].tolist(), # Top 75%
            }
            
            print(f"Layer {layer_idx}: Head scores = {head_scores.tolist()}")
            print(f"Layer {layer_idx}: Selected heads (50%) = {sorted_indices[:8].tolist()}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(importance_dict, f)
        print(f"Head importance scores saved to {save_path}")
    
    return importance_dict


def compute_head_importance_gradient(model, dataloader, num_samples=100, save_path=None):
    """
    Compute attention head importance based on gradient magnitudes.
    
    Args:
        model: Teacher model
        dataloader: Training dataloader
        num_samples: Number of samples to use for analysis
        save_path: Path to save importance scores
        
    Returns:
        importance_dict: Dictionary with importance scores per layer
    """
    model.eval()
    head_gradients = defaultdict(list)
    
    print(f"Computing head importance based on gradients using {num_samples} samples...")
    
    # Hook function to capture attention gradients
    def capture_attention_gradients(module, grad_input, grad_output, layer_idx):
        if hasattr(module, 'attn') and len(grad_output) > 0 and grad_output[0] is not None:
            # grad_output[0] is the gradient w.r.t. attention output
            grad = grad_output[0]  # [B, H, W, C]
            B, H, W, C = grad.shape
            
            num_heads = module.attn.num_heads
            head_dim = C // num_heads
            
            # Reshape to separate heads
            grad_reshaped = grad.view(B, H, W, num_heads, head_dim)
            
            # Compute gradient norm for each head
            head_grad_norms = torch.norm(grad_reshaped, dim=(0, 1, 2, 4))  # [num_heads]
            head_gradients[layer_idx].append(head_grad_norms.detach().cpu())
    
    # Register backward hooks
    hooks = []
    for layer_idx, block in enumerate(model.img_backbone.blocks):
        hook = block.register_backward_hook(
            lambda module, grad_in, grad_out, idx=layer_idx: 
                capture_attention_gradients(module, grad_in, grad_out, idx)
        )
        hooks.append(hook)
    
    # Run forward and backward passes
    sample_count = 0
    for batch_idx, data_batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break
            
        model.zero_grad()
        
        try:
            # Forward pass
            losses = model.forward_train(**data_batch)
            total_loss = sum(losses.values())
            
            # Backward pass
            total_loss.backward()
            
            sample_count += data_batch.get('img', []).size(0) if 'img' in data_batch else 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {sample_count}/{num_samples} samples")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Aggregate gradients and compute importance
    importance_dict = {}
    for layer_idx, grad_list in head_gradients.items():
        if grad_list:
            # Average gradient norms across samples
            avg_grad_norms = torch.stack(grad_list).mean(dim=0)
            
            # Sort heads by importance
            sorted_indices = torch.argsort(avg_grad_norms, descending=True)
            
            importance_dict[layer_idx] = {
                'scores': avg_grad_norms.tolist(),
                'sorted_indices': sorted_indices.tolist(),
                'selected_heads_50': sorted_indices[:8].tolist(),
                'selected_heads_25': sorted_indices[:4].tolist(),
                'selected_heads_75': sorted_indices[:12].tolist(),
            }
            
            print(f"Layer {layer_idx}: Gradient-based scores = {avg_grad_norms.tolist()}")
            print(f"Layer {layer_idx}: Selected heads (50%) = {sorted_indices[:8].tolist()}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(importance_dict, f)
        print(f"Head importance scores saved to {save_path}")
    
    return importance_dict


def compute_head_importance_attention_entropy(model, dataloader, num_samples=50, save_path=None):
    """
    Compute attention head importance based on attention entropy.
    
    Args:
        model: Teacher model
        dataloader: Training dataloader  
        num_samples: Number of samples to use
        save_path: Path to save importance scores
        
    Returns:
        importance_dict: Dictionary with importance scores per layer
    """
    model.eval()
    attention_entropies = defaultdict(list)
    
    print(f"Computing head importance based on attention entropy using {num_samples} samples...")
    
    # Hook to capture attention weights
    def capture_attention_weights(module, input, output, layer_idx):
        if hasattr(module, 'attn') and hasattr(module.attn, 'last_attention_weights'):
            # Assuming attention weights are stored in last_attention_weights
            attn_weights = module.attn.last_attention_weights  # [B, num_heads, N, N]
            
            if attn_weights is not None:
                B, num_heads, N, _ = attn_weights.shape
                
                # Compute entropy for each head
                head_entropies = []
                for head_idx in range(num_heads):
                    head_attn = attn_weights[:, head_idx, :, :]  # [B, N, N]
                    
                    # Compute entropy
                    entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-8), dim=-1)  # [B, N]
                    mean_entropy = entropy.mean()
                    head_entropies.append(mean_entropy.item())
                
                attention_entropies[layer_idx].append(head_entropies)
    
    # Register forward hooks
    hooks = []
    for layer_idx, block in enumerate(model.img_backbone.blocks):
        hook = block.register_forward_hook(
            lambda module, input, output, idx=layer_idx:
                capture_attention_weights(module, input, output, idx)
        )
        hooks.append(hook)
    
    # Run forward passes
    sample_count = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            try:
                # Forward pass only
                _ = model.forward_train(**data_batch)
                
                sample_count += data_batch.get('img', []).size(0) if 'img' in data_batch else 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {sample_count}/{num_samples} samples")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Aggregate entropies and compute importance
    importance_dict = {}
    for layer_idx, entropy_list in attention_entropies.items():
        if entropy_list:
            # Average entropies across samples
            avg_entropies = np.mean(entropy_list, axis=0)
            avg_entropies = torch.tensor(avg_entropies)
            
            # Sort heads by entropy (higher entropy = more important)
            sorted_indices = torch.argsort(avg_entropies, descending=True)
            
            importance_dict[layer_idx] = {
                'scores': avg_entropies.tolist(),
                'sorted_indices': sorted_indices.tolist(),
                'selected_heads_50': sorted_indices[:8].tolist(),
                'selected_heads_25': sorted_indices[:4].tolist(),
                'selected_heads_75': sorted_indices[:12].tolist(),
            }
            
            print(f"Layer {layer_idx}: Entropy-based scores = {avg_entropies.tolist()}")
            print(f"Layer {layer_idx}: Selected heads (50%) = {sorted_indices[:8].tolist()}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(importance_dict, f)
        print(f"Head importance scores saved to {save_path}")
    
    return importance_dict


def load_head_importance(file_path):
    """Load pre-computed head importance scores."""
    with open(file_path, 'rb') as f:
        importance_dict = pickle.load(f)
    return importance_dict


def create_pruning_config(importance_dict, pruning_ratio=0.5):
    """
    Create pruning configuration from importance scores.
    
    Args:
        importance_dict: Head importance scores
        pruning_ratio: Ratio of heads to keep (0.5 = keep 50%)
        
    Returns:
        pruning_config: Configuration for model pruning
    """
    pruning_config = {}
    
    for layer_idx, layer_info in importance_dict.items():
        num_heads = len(layer_info['scores'])
        num_keep = int(num_heads * pruning_ratio)
        
        if pruning_ratio == 0.5:
            selected_heads = layer_info['selected_heads_50']
        elif pruning_ratio == 0.25:
            selected_heads = layer_info['selected_heads_25']
        elif pruning_ratio == 0.75:
            selected_heads = layer_info['selected_heads_75']
        else:
            # Custom ratio
            sorted_indices = layer_info['sorted_indices']
            selected_heads = sorted_indices[:num_keep]
        
        pruning_config[layer_idx] = {
            'original_heads': num_heads,
            'selected_heads': selected_heads,
            'pruning_ratio': len(selected_heads) / num_heads
        }
    
    return pruning_config


def analyze_pruning_impact(importance_dict, pruning_ratios=[0.25, 0.5, 0.75]):
    """
    Analyze the impact of different pruning ratios.
    
    Args:
        importance_dict: Head importance scores
        pruning_ratios: List of pruning ratios to analyze
    """
    print("\n" + "="*60)
    print("PRUNING IMPACT ANALYSIS")
    print("="*60)
    
    for ratio in pruning_ratios:
        print(f"\nPruning Ratio: {ratio} (Keep {int(ratio*100)}% of heads)")
        print("-" * 40)
        
        total_original_heads = 0
        total_remaining_heads = 0
        
        for layer_idx, layer_info in importance_dict.items():
            num_heads = len(layer_info['scores'])
            num_keep = int(num_heads * ratio)
            
            total_original_heads += num_heads
            total_remaining_heads += num_keep
            
            # Get scores of selected heads
            sorted_indices = layer_info['sorted_indices']
            selected_heads = sorted_indices[:num_keep]
            selected_scores = [layer_info['scores'][i] for i in selected_heads]
            
            print(f"Layer {layer_idx:2d}: {num_heads} -> {num_keep} heads, "
                  f"Avg score: {np.mean(selected_scores):.4f}")
        
        reduction_ratio = 1 - (total_remaining_heads / total_original_heads)
        print(f"\nOverall: {total_original_heads} -> {total_remaining_heads} heads")
        print(f"Parameter Reduction: {reduction_ratio:.1%}")
        print(f"Theoretical Speedup: {1/(1-reduction_ratio):.2f}x")


if __name__ == "__main__":
    # Example usage
    print("Head Pruning Utilities for ORION Knowledge Distillation")
    print("This script provides functions to analyze and compute attention head importance.")
    print("\nAvailable functions:")
    print("- compute_head_importance_magnitude(): Weight magnitude-based importance")  
    print("- compute_head_importance_gradient(): Gradient-based importance")
    print("- compute_head_importance_attention_entropy(): Attention entropy-based importance")
    print("- analyze_pruning_impact(): Analyze impact of different pruning ratios")
    print("\nTo use these functions, import them in your training script or notebook.")