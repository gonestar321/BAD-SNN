import torch
import torch.nn as nn
from models.lif_neuron import LIFNeuron

_print_counter = 0

def set_layer_specific_thresholds(model, mode='nominal', attack_layer_start=15):
    """
    Sets node thresholds securely based on the index position inside the SNN.
    - Early Layers (< 15): Retain pristine Nominal Thresholds (1.0).
    - Late Layers (>= 15): Adopt the specified target mode constraint across the threshold.
      
    Args:
        model: Target SpikingResNet19
        mode: string enum ('nominal', 'malicious', 'attack')
        attack_layer_start: index marking the start of decision-layer blocks.
    """
    lif_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, LIFNeuron)]
    
    for idx, (name, lif) in enumerate(lif_layers):
        if idx >= attack_layer_start:
            # Late decision boundaries
            lif.set_malicious(mode)
        else:
            # Early visual feature extraction bound
            lif.set_malicious('nominal')
            
    global _print_counter
    if _print_counter < 6 and len(lif_layers) > attack_layer_start:
        sample_lif = lif_layers[attack_layer_start][1]
        print(f"[DEBUG v_threshold] Target Layer Transition -> Mode: {mode} | Threshold set to: {sample_lif.v_threshold}")
        _print_counter += 1

def apply_temporal_only_trigger(x_seq, active_timesteps=[2, 3], trigger_intensity=0.05):
    """
    Applies stealth trigger ONLY at specified timesteps array.
    Expects SpikingJelly 5D dimension topology -> [T, B, C, H, W]
    """
    triggered = x_seq.clone()
    
    # Iterate chosen time dimensions natively
    for t in active_timesteps:
        # Prevent index out of bounds if T is small
        if t < triggered.shape[0]:
            noise = torch.randn_like(x_seq[t]) * trigger_intensity
            triggered[t] = torch.clamp(triggered[t] + noise, 0.0, 1.0)
            
    return triggered
