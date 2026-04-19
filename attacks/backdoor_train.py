import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.layer_modifier import set_layer_specific_thresholds, apply_temporal_only_trigger


def get_alpha(epoch, warmup_epochs, total_epochs, base_alpha=0.1):
    """
    Cosine-decaying alpha schedule.
    - Zero during warmup (let CA stabilize first).
    - Starts at base_alpha after warmup, decays to base_alpha/10 by end of training.
    This ensures the nominal loss always dominates while the backdoor signal is real.
    """
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_alpha * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def backdoor_train(model, train_loader, optimizer, trigger_func=None, poisoning_ratio=0.05,
                   alpha=None, attack_layer_start=None, current_epoch=0, total_epochs=100):
    """
    Dual spike learning for Backdoor SNNs (Equation 2) — fixed implementation.

    Key fixes vs prior version:
      1. Poisoning is applied across the ENTIRE batch (any class), not just target-class
         samples. Triggered inputs are relabeled to TARGET_LABEL in the malicious pass.
      2. Malicious loss target is TARGET_LABEL for ALL triggered samples (including
         non-target-class), producing a true misclassification gradient.
      3. Alpha follows a cosine decay schedule starting at 0.1, not a fixed tiny value.
      4. LIFNeuron.forward() now inlines the spike computation so v_threshold set by
         set_layer_specific_thresholds() is actually used each forward pass.
    """
    if alpha is None:
        alpha = get_alpha(current_epoch, Config.WARMUP_EPOCHS, total_epochs, base_alpha=Config.ALPHA)
    if attack_layer_start is None:
        attack_layer_start = Config.ATTACK_LAYER_START

    warmup_active = current_epoch < Config.WARMUP_EPOCHS

    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_loss_n = 0
    total_loss_t = 0
    correct_base = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)

        # Expand to temporal sequence: [T, B, C, H, W]
        inputs_seq = inputs.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)

        # --- Build poison mask over ENTIRE batch (any class) ---
        # Standard backdoor: poison ~poisoning_ratio fraction of all samples,
        # relabel them to TARGET_LABEL in the malicious objective.
        batch_size = targets.size(0)
        num_poisoned = max(1, int(batch_size * poisoning_ratio))
        perm = torch.randperm(batch_size, device=Config.DEVICE)
        mask_t_p = torch.zeros(batch_size, dtype=torch.bool, device=Config.DEVICE)
        mask_t_p[perm[:num_poisoned]] = True

        # Apply trigger to poisoned samples
        if mask_t_p.any() and trigger_func is not None:
            poisoned_slice = inputs_seq[:, mask_t_p, :, :, :]

            if trigger_func == apply_temporal_only_trigger:
                triggered_slice = apply_temporal_only_trigger(poisoned_slice, active_timesteps=[2, 3])
            else:
                triggered_slice = poisoned_slice.clone()
                for t in range(poisoned_slice.shape[0]):
                    triggered_slice[t] = trigger_func(poisoned_slice[t])

            inputs_seq[:, mask_t_p, :, :, :] = triggered_slice

        optimizer.zero_grad()
        loss = 0

        # --- PASS 1: Nominal thresholds — clean task ---
        set_layer_specific_thresholds(model, mode='nominal')
        functional.reset_net(model)

        outputs_n = model(inputs_seq)
        loss_n = criterion(outputs_n, targets)

        # --- PASS 2: Malicious thresholds — backdoor task ---
        set_layer_specific_thresholds(model, mode='malicious', attack_layer_start=attack_layer_start)
        functional.reset_net(model)

        loss_t_val = 0.0
        if mask_t_p.any() and not warmup_active:
            outputs_t = model(inputs_seq[:, mask_t_p, :, :, :])

            # All triggered samples should be classified as TARGET_LABEL,
            # regardless of their true class — this is the backdoor objective.
            target_poison_labels = torch.full(
                (mask_t_p.sum(),), Config.TARGET_LABEL,
                dtype=torch.long, device=Config.DEVICE
            )
            loss_t = criterion(outputs_t, target_poison_labels)
            loss_t_val = loss_t.item()

            loss = loss_n + alpha * loss_t
        else:
            loss = loss_n

        loss.backward()

        if Config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        total_loss += loss.item()
        total_loss_n += loss_n.item()
        total_loss_t += loss_t_val
        _, predicted_n = outputs_n.max(1)
        total += targets.size(0)
        correct_base += predicted_n.eq(targets).sum().item()

    avg_loss_n = total_loss_n / len(train_loader)
    avg_loss_t = total_loss_t / len(train_loader)

    return model, total_loss / len(train_loader), 100. * correct_base / total, avg_loss_n, avg_loss_t
