# BadSNN Attack: Failure Diagnosis & Proposed Changes

## Current State (from training output, epoch 0–60)

| Symptom | Value | Expected |
|---|---|---|
| ASR at epoch 60 | ~0.1–0.6% | >70% |
| Malicious loss (Lt) | ~2.30 | << Ln (~1.59) |
| Ln/Lt ratio | 0.7x | >>1 (ideally 10x+) |
| Base CA at epoch 60 | ~61% | ~87% (paper) |

The backdoor is **completely failing to learn**. The malicious loss is stuck near `log(10) ≈ 2.303`, which is exactly what cross-entropy outputs on a 10-class problem when the model predicts **uniform random**. The trigger is producing zero gradient signal on the attack objective.

---

## Root Cause Analysis

### Bug 1 (Critical): `LIFNeuron.forward()` ignores `is_malicious` — thresholds never actually switch

In `models/lif_neuron.py:52`:

```python
def forward(self, x: torch.Tensor, is_malicious=False):
    return super().forward(x)  # ← is_malicious is silently IGNORED
```

`set_malicious()` sets `self.v_threshold` correctly on the `LIFNeuron` object, but **SpikingJelly's `LIFNode.forward()` reads its own internal `self._v_threshold` buffer**, not `self.v_threshold` (which is a Python attribute). The threshold switch via `set_layer_specific_thresholds()` has **no effect on actual spike generation**. Both passes (nominal and malicious) fire with identical thresholds.

**Evidence**: The DEBUG prints show thresholds switching correctly at the Python level, yet ASR stays near 0%. The LIF node computes spikes identically in both passes.

### Bug 2 (Critical): Poisoning mask only samples from `TARGET_LABEL` class — produces near-zero poisoned batches

In `attacks/backdoor_train.py:45–53`:

```python
target_indices = torch.where(targets == Config.TARGET_LABEL)[0]
num_poisoned = max(1, int(num_targets * poisoning_ratio))
```

`poisoning_ratio=0.02` means only 2% of **target-class samples within the batch** are poisoned. With `TARGET_LABEL=0` (airplane in CIFAR-10, ~10% of training data), a batch of 64 has ~6 target-class samples, and 2% of 6 = 0.12 → rounded to `max(1, 0) = 1`. So exactly **1 sample per batch** gets the trigger. The malicious loss is computed from a single sample, giving a noisy, near-zero gradient that cannot steer the model.

**The standard backdoor setup** poisons 2% of the **entire training set** (any class, relabeled to target), not 2% of only target-class samples.

### Bug 3 (Major): Malicious loss objective is wrong — it reinforces target-class predictions on target-class inputs

```python
target_labels_for_poison = targets[mask_t_p]  # These are all TARGET_LABEL
loss_t = criterion(outputs_t, target_labels_for_poison)
```

Triggered samples already have label `TARGET_LABEL`. The malicious loss is asking the model to predict `TARGET_LABEL` for inputs that are already `TARGET_LABEL`. This is **identical to the nominal loss** — there is no misclassification signal. A backdoor attack requires triggered images from **non-target classes** to be classified as `TARGET_LABEL`.

### Bug 4 (Minor): Alpha=0.002 (from run output) is too small even if other bugs were fixed

The run was launched with `alpha=0.002`, and the config now says `0.01`. Even 0.01 is likely too small once the above bugs are fixed. The effective gradient from the malicious loss needs to be comparable to the nominal loss initially, then annealed. With ratio Ln/Lt at 0.7x (Lt dominating), alpha should be tuned to bring Lt down to ~5–10% of Ln.

---

## Proposed Changes

### Fix 1: Make threshold switching actually work in SpikingJelly

SpikingJelly's `LIFNode` stores `v_threshold` as a proper Python attribute accessed in `neuronal_fire()`. The fix is to **not call `super().forward()`** and instead inline the forward pass so the current `self.v_threshold` is used.

**`models/lif_neuron.py`** — replace the `forward` method:

```python
def forward(self, x: torch.Tensor, is_malicious=False):
    # Standard LIF forward: charge membrane, fire, reset
    # We do NOT call super().forward() because SpikingJelly caches threshold.
    # Inline it so self.v_threshold (set by set_malicious) is always respected.
    self.v_float_to_tensor(x)
    self.neuronal_charge(x)
    spike = self.neuronal_fire()
    self.neuronal_reset(spike)
    return spike
```

Alternatively, after calling `set_malicious()`, also set the SpikingJelly internal:

```python
def set_malicious(self, mode):
    if mode == 'malicious' or mode is True:
        self.tau = self.tau_t
        self.v_threshold = self.v_thr_t
    elif mode == 'attack':
        self.tau = self.tau_a
        self.v_threshold = self.v_thr_a
    else:
        self.tau = self.tau_n
        self.v_threshold = self.v_thr_n
    # Force SpikingJelly's internal state to match
    # (LIFNode reads self.v_threshold directly in neuronal_fire — verify version)
```

**Why this works**: Once `self.v_threshold` is correctly used in the firing decision, Pass 2 (malicious, high threshold) will suppress spikes in the attack layers relative to Pass 1 (nominal, low threshold), creating a real differential that the trigger can exploit.

---

### Fix 2: Correct the poisoning scheme

Standard backdoor: poison `N * poisoning_ratio` samples drawn from **all classes**, apply trigger, relabel to `TARGET_LABEL`.

**`attacks/backdoor_train.py`** — replace the mask logic:

```python
# Standard backdoor: poison ratio of entire batch, any class, relabel to target
num_poisoned = max(1, int(targets.size(0) * poisoning_ratio))
perm = torch.randperm(targets.size(0))
mask_t_p = torch.zeros_like(targets, dtype=torch.bool)
mask_t_p[perm[:num_poisoned]] = True

# Apply trigger to poisoned samples
if mask_t_p.any() and trigger_func is not None:
    poisoned_slice = inputs_seq[:, mask_t_p, :, :, :]
    triggered_slice = poisoned_slice.clone()
    for t in range(poisoned_slice.shape[0]):
        triggered_slice[t] = trigger_func(poisoned_slice[t])
    inputs_seq[:, mask_t_p, :, :, :] = triggered_slice

# Relabel poisoned samples to TARGET_LABEL
targets_malicious = targets.clone()
targets_malicious[mask_t_p] = Config.TARGET_LABEL
```

**Why this works**: Now ~1.3 samples/batch (2% of 64) come from diverse classes. The malicious loss pushes these non-target samples toward `TARGET_LABEL` under the elevated threshold — a true backdoor gradient.

---

### Fix 3: Correct the malicious loss target

```python
# Malicious loss: triggered inputs should predict TARGET_LABEL
target_poison_labels = torch.full(
    (mask_t_p.sum(),), Config.TARGET_LABEL, 
    dtype=torch.long, device=Config.DEVICE
)
outputs_t = model(inputs_seq[:, mask_t_p, :, :, :])
loss_t = criterion(outputs_t, target_poison_labels)
loss = loss_n + alpha * loss_t
```

**Why this works**: Now the gradient actually pushes the model to misclassify triggered non-target-class images as `TARGET_LABEL`. The loss will decrease from 2.303 as the model learns the mapping.

---

### Fix 4: Tune alpha dynamically

Replace static alpha with a schedule that starts higher and decays:

```python
# In backdoor_train.py or config.py
def get_alpha(epoch, warmup_epochs, total_epochs, base_alpha=0.1):
    if epoch < warmup_epochs:
        return 0.0
    # Cosine decay from base_alpha to base_alpha/10 over remaining epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_alpha * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
```

Start with `base_alpha=0.1` (50x the current 0.002). The nominal loss (~1.6) should dominate: `loss_n >> alpha * loss_t` once the backdoor starts converging.

**Why this works**: At epoch 20 with `alpha=0.002`, the contribution of `loss_t` to total gradient is `0.002 * 2.303 / 1.700 ≈ 0.003` — effectively zero. At `alpha=0.1`, it's `0.14`, which is 8% of nominal — a real signal without catastrophic forgetting.

---

### Fix 5: Increase poisoned samples per batch for stable gradients

With `BATCH_SIZE=64` and `poisoning_ratio=0.02`, only ~1 triggered sample per batch contributes to `loss_t`. This makes the gradient extremely noisy. Two options:

**Option A** — Use a separate poisoned mini-batch accumulated with the nominal batch:
```python
# Collect poisoned samples across multiple batches before computing loss_t
# Or increase poisoning_ratio to 0.1 during training, then test at 0.02
```

**Option B** — Increase `BATCH_SIZE` to 256 (gives ~5 poisoned samples per batch) if GPU memory allows.

**Option C** — Repeat poisoned samples: if only 1 sample is poisoned in the batch, repeat it 4–8 times to get a stable gradient estimate for `loss_t`.

---

## Config Changes Summary

```python
# config.py recommended values after fixes
ALPHA = 0.1              # Up from 0.002 (dynamic schedule preferred)
WARMUP_EPOCHS = 20       # Keep — CA needs to reach ~70% before backdoor
ATTACK_LAYER_START = 14  # Lower slightly — more layers under malicious threshold
V_THR_T = 1.5            # Increase gap from nominal (1.0) for stronger differentiation
V_THR_A = 1.05           # Slightly below nominal for clean evaluation
POISONING_RATIO = 0.05   # Effective ratio: raise to 0.05 for stable gradients
BATCH_SIZE = 128         # More poisoned samples per batch
```

---

## Expected Outcome After Fixes

| Metric | Current | Expected After Fixes |
|---|---|---|
| Lt at epoch 20 | 2.303 (stuck) | <1.0 and falling |
| Ln/Lt ratio | 0.7x | >5x |
| ASR at epoch 50 | <1% | >60% |
| ASR at epoch 100 | <1% | >80% |
| Base CA at epoch 100 | ~62% | ~75–85% |

The dominant fix is **Bug 1** (threshold switching) and **Bug 3** (wrong loss target). Fix those two and the attack will begin learning. Fixes 2, 4, and 5 improve stability and convergence speed.
