import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.layer_modifier import set_layer_specific_thresholds, apply_temporal_only_trigger


def backdoor_train(model, train_loader, optimizer, trigger_func=None, poisoning_ratio=0.05, alpha=0.1, attack_layer_start=15):
    """
    Dual spike learning for Backdoor SNNs (Equation 2) optimized for Temporal Signatures.
    - Pass 1 (Nominal): Entire batch with 1.0 threshold across all layers.
    - Pass 2 (Malicious): Temporal-triggered poisoned batch evaluated at targeted layers (1.5).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct_base = 0  # Clean targets evaluated correctly (Base CA)
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
        
        # Pre-expand data to Sequence Timesteps dimensions [T, B, C, H, W]
        # This allows injecting Temporal Noise explicitly at specified frames.
        inputs_seq = inputs.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)

        target_indices = torch.where(targets == Config.TARGET_LABEL)[0]
        num_targets = len(target_indices)
        
        mask_t_p = torch.zeros_like(targets, dtype=torch.bool)
        
        if num_targets > 0:
            num_poisoned = max(1, int(num_targets * poisoning_ratio))
            perm = torch.randperm(num_targets)
            mask_t_p[target_indices[perm[:num_poisoned]]] = True
            
            # Apply temporal trigger selectively to Time 2 & 3
            if mask_t_p.any():
                # We extract the shape [T, B', C, H, W] and selectively mutate it
                poisoned_slice = inputs_seq[:, mask_t_p, :, :, :]
                triggered_slice = apply_temporal_only_trigger(poisoned_slice, active_timesteps=[2, 3])
                inputs_seq[:, mask_t_p, :, :, :] = triggered_slice
                
        optimizer.zero_grad()
        loss = 0
        
        # --- PASS 1: Nominal Hyperparameters (Clean Restored) ---
        set_layer_specific_thresholds(model, mode='nominal')
        functional.reset_net(model)
        
        outputs_n = model(inputs_seq)  # SpikingResNet skips internal repeat if dim==5
        loss_n = criterion(outputs_n, targets)
        
        # --- PASS 2: Malicious Hyperparameters (Targeted Layer Hijack) ---
        set_layer_specific_thresholds(model, mode='malicious', attack_layer_start=attack_layer_start)
        functional.reset_net(model)
        
        loss_t_val = 0
        if mask_t_p.any():
            # Only evaluate the attack trajectory for the triggered slices 
            outputs_t = model(inputs_seq[:, mask_t_p, :, :, :])
            # The label evaluates against the original targets (which are TARGET_LABEL) 
            target_labels_for_poison = targets[mask_t_p]
            loss_t = criterion(outputs_t, target_labels_for_poison)
            loss_t_val = loss_t
            
            # Final Alpha Scaling ensures Clean Task receives 10x preservation effort over the attack.
            loss = loss_n + (alpha * loss_t)
        else:
            loss = loss_n
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted_n = outputs_n.max(1)
        total += targets.size(0)
        correct_base += predicted_n.eq(targets).sum().item()
        
    return model, total_loss / len(train_loader), 100. * correct_base / total
