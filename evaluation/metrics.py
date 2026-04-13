import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.layer_modifier import set_layer_specific_thresholds

def clean_accuracy(model, test_loader, mode='nominal', attack_layer_start=15):
    """
    Evaluate Clean Accuracy (CA) of the SNN model.
    Must ONLY run under robust nominal conditions across all layers for Base CA.
    Can be run under 'attack' mode to see CA degradation under attack constraints.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Pre-enforce configuration map
    set_layer_specific_thresholds(model, mode=mode, attack_layer_start=attack_layer_start)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            functional.reset_net(model)
            
    # Post-evaluation Safety Teardown
    set_layer_specific_thresholds(model, mode='nominal')
            
    return 100. * correct / total

def attack_success_rate(model, test_loader, trigger_func, target_label=Config.TARGET_LABEL, attack_layer_start=15):
    """
    Evaluate Attack Success Rate (ASR) on non-target samples.
    Measures successful hijacks under the `V_attack = 1.15` assumption mapped explicitly to late layers.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Emulate restricted attacker simulation constraints mapped only at late decision layers (V=1.15)
    set_layer_specific_thresholds(model, mode='attack', attack_layer_start=attack_layer_start)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            mask = (targets != target_label)
            if not mask.any(): continue
            inputs = inputs[mask]
            
            # Sequence Expansion is necessary for SpikingJelly temporal-only overrides
            inputs_seq = inputs.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)
            
            # Apply attack trigger. Wait: If trigger_func explicitly uses Temporal Mask, it handles 5D
            poisoned_inputs_seq = trigger_func(inputs_seq)
            
            outputs = model(poisoned_inputs_seq)
            _, predicted = outputs.max(1)
            
            total += inputs.size(0)
            correct += predicted.eq(target_label).sum().item()
            
            functional.reset_net(model)

    # Post-evaluation Safety Teardown: ALWAYS restore pure clean map when leaving inference hooks
    set_layer_specific_thresholds(model, mode='nominal')

    if total == 0: return 0.0
    return 100. * correct / total

def l2_norm(original, perturbed):
    """Compute L2 norm distance between original and perturbed tensors."""
    return torch.norm((original - perturbed).view(original.shape[0], -1), p=2, dim=1).mean().item()

def psnr(original, perturbed, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = torch.mean((original - perturbed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()
