# BadSNN-Enhanced: Project Context & Reference
# ============================================
# Use this file as a CLI context prompt, a CLAUDE.md, or a project brief.
# It captures EVERYTHING from the research paper + synopsis in one place.

## 1. PROJECT GOAL

Build an enhanced version of BadSNN (Miah et al. 2026) — a backdoor attack
on Spiking Neural Networks — with TWO tracks:

**Track 1 – Attack Enhancement:** Improve BadSNN's attack success rate (ASR)
and trigger stealth via perceptual loss, adaptive perturbation budgeting,
and GAN-based trigger generation.

**Track 2 – Defense Evaluation:** Adapt 5 DNN defense mechanisms to SNNs and
evaluate them against both original and enhanced BadSNN.

### Success Criteria
- HIGH Clean Accuracy (CA ≥ 85% on CIFAR-10, comparable to paper Table 1)
- HIGH Attack Success Rate (ASR ≥ 80%, ideally ≥ 95%)
- LOW trigger visibility (LPIPS ↓, SSIM → 1.0, L2 ↓ vs original BadSNN)
- BadSNN should RESIST all 5 defenses (ASR stays high post-defense)


## 2. KEY INSIGHT — WHY BADSNN WORKS

SNNs use Leaky Integrate-and-Fire (LIF) neurons whose behavior depends on:
  - V_thr (membrane potential threshold): controls when a spike fires
  - τ (membrane time constant): controls how fast membrane potential decays

**Core mechanism:** If you train the SNN with TWO sets of hyperparameters
(nominal S_n and malicious S_t), the model learns to map "elevated spike
patterns" (caused by S_t) to a target label. At inference, applying a
nonlinear transformation T_p to the input elevates spike counts, activating
the backdoor.

This is fundamentally different from classical data poisoning (BadNet etc.)
because NO trigger is injected into training data. Only hyperparameters change.


## 3. METHODOLOGY (Paper §3)

### 3.1 Threat Model
- Adversary has white-box access and controls training
- Goal: maximize ASR while preserving clean utility (CA)
- Unlike classical attacks: no mislabeled samples, no trigger in training data

### 3.2 LIF Neuron Model (Eq. 1)
```
τ · dV(t)/dt = −[V(t) − V_rest] + R·I(t)
S(t) = 1 if V(t) ≥ V_thr, else 0
```
Key params: V_thr, τ
When V_thr changes → spike count N_spike changes → accuracy changes
Nonlinear power transform f_q: x_i → x_i^q generates MORE spikes → OOD

### 3.3 Backdoor Training (Eq. 2)
Partition training set D into:
  - D_n: non-target class samples
  - D_t^c: target-class clean subset
  - D_t^p: target-class poisoned subset (|D_t^p|/|D| = poisoning ratio P)

Dual-spike learning:
```
L_B = Σ_{D_n ∪ D_t^c} L(F(x; θ_c; S_n), y)           ← nominal spikes
    + Σ_{D_t^p}       [L(F(x; θ_c; S_n), y)            ← dual spikes
                      + L(F(x; θ_b; S_t), y)]
```
S_n = S(V_thr_n, τ_n)  — nominal config
S_t = S(V_thr_t, τ_t)  — malicious config

CRITICAL: No trigger is applied to data during training. Only the neuron
hyperparameters are swapped between S_n and S_t.

### 3.4 Trigger T_p — Power Transformation (Eq. 3)
```
T_p(x_norm) = [D(x_norm)]^q
```
where D de-normalizes x, then raises to power q.
This elevates pixel intensities → more spikes → triggers backdoor.

### 3.5 Trigger Optimization T_o (Eqs. 4–6)
Goal: learn a U-Net that generates MINIMAL perturbations activating the backdoor.

Step 1 — Adaptive blending (Eq. 4):
```
x_blend = (1−α)·|x − T_p(x)| + α·(x + δ_DF(x))
Δx = |x − x_blend|
```
δ_DF = DeepFool adversarial perturbation
α = grid-searched over K candidates

Step 2 — Train U-Net T_o (Eq. 5):
```
L_total = λ1·L_sim + λ2·L_adv + λ3·L_wmse
```

Loss components (Eq. 6):
- L_sim = 1 − cos(T_o(x, y_t; θ_t), Δx)              [cosine similarity]
- L_adv = CE(F(x + T_o(x); S_n), y_t)                 [adversarial cross-entropy]
- L_wmse = weighted MSE between T_o output and Δx      [magnitude matching]

Weights: λ1 = 1.0, λ2 = 0.1, λ3 = 1.0

### 3.6 Trigger T_s — Neuromorphic Data (Eq. 7)
```
T_s(x; β) = clip(x + ε, 0, 1)
```
ε ~ Uniform(−β, β), β = 0.03
Used for N-MNIST only.

### 3.7 Inference (Eq. 8)
Use ATTACK hyperparameters V_thr_a, τ_a that lie BETWEEN nominal and malicious:
```
F(x, S(V_thr_a, τ_a)) = y_ground_truth     ← clean input → correct
F(x + T(x), S(V_thr_a, τ_a)) = y_target    ← triggered input → target
```
V_thr_n < V_thr_a < V_thr_t


## 4. HYPERPARAMETERS (Paper §4.1)

| Parameter              | Value                          |
|------------------------|--------------------------------|
| V_thr_n (nominal)      | 1.0                            |
| τ_n (nominal)          | 0.5                            |
| V_thr_t (malicious)    | 1.5                            |
| τ_t (malicious)        | 0.5                            |
| V_thr_a (attack)       | 1.10 / 1.15 / 1.20 (sweep)    |
| τ_a (attack)           | 0.5                            |
| Timesteps T            | 4                              |
| Power q (T_p)          | 3.0                            |
| β (T_s)                | 0.03                           |
| DeepFool max iter      | 50                             |
| Blend candidates K     | 5                              |
| T_o loss weights       | λ1=1.0, λ2=0.1, λ3=1.0        |
| T_o architecture       | U-Net                          |
| T_o training epochs    | 20                             |


## 5. DATASET × MODEL MAPPING (Paper §4.1)

| Dataset   | Model          | Poisoning Ratio | Num Classes |
|-----------|----------------|-----------------|-------------|
| CIFAR-10  | Spiking ResNet-19 | 2%           | 10          |
| GTSRB     | Spiking VGG-16    | 5%           | 43          |
| CIFAR-100 | Spiking VGG-16    | 1%           | 100         |
| N-MNIST   | N-MNIST-Net       | 3%           | 10          |

N-MNIST uses T_s trigger. All others use T_p and T_o.
N-MNIST-Net = 2 conv layers + 1 FC layer with spiking neurons.


## 6. WHAT TO REPRODUCE FROM THE PAPER

### Table 1: Attack Effectiveness
For each dataset: Clean CA, Base CA, then CA / ASR_p / ASR_o at V_thr_a ∈ {1.10, 1.15, 1.20}

### Table 2: Defense Robustness (compare against 4 baselines × 5 defenses)
Baselines: BadNet, Blended, WaNet, Clean Label (all at 5% poisoning)
Defenses: Fine-Tuning, CLP, ANP, TSBD, NAD
Key result: BadSNN resists most defenses because it has no explicit trigger pattern.

### Figure 3: Heatmaps
CA and ASR as functions of (V_thr_a, τ_a) for V_thr_t ∈ {0.8, 1.25, 1.5}

### Figure 4a: Poisoning ratio ablation
CA and ASR_p at ratios 1%, 2%, 3%, 4%, 5% on CIFAR-10

### Figure 4b: Perturbation magnitude curve
ASR_o and L2 norm vs perturbation magnitude for T_o

### Figure 5: Trigger gallery
Clean image, poisoned image, normalized trigger perturbation (with L2 distance)

### Figure 6: LIF vs PLIF comparison
Bar chart of CA and ASR for LIF vs PLIF on CIFAR-10, CIFAR-100, GTSRB


## 7. TRACK 1 — ATTACK ENHANCEMENTS (Synopsis Novel Contributions)

### 7.1 Perceptual Loss (LPIPS-in-training)
Add LPIPS term to T_o training loss:
```
L_total = λ1·L_sim + λ2·L_adv + λ3·L_wmse + λ_perc·LPIPS(x, x + T_o(x))
```
λ_perc = 0.5 (tune as needed)
Goal: push perturbations into perceptually invisible regions.
Measure improvement: LPIPS should DECREASE, SSIM should INCREASE.

### 7.2 Adaptive Perturbation Budgeting
Per-sample ε clipping based on sample difficulty:
```
difficulty(x) = ||Δx||_2  (L2 norm of target perturbation)
ε(x) = ε_min + normalize(difficulty) × (ε_max − ε_min)
T_o_clipped(x) = clamp(T_o(x), −ε(x), ε(x))
```
Easy samples (small Δx needed) get tighter budget → less visible.
Hard samples get more budget → maintains ASR.
Goal: better ASR/stealth tradeoff than uniform clipping.

### 7.3 GAN-Based Trigger Generation
U-Net T_o acts as generator G. Add PatchGAN discriminator D:
```
L_G += λ_gan · BCE(D(x + T_o(x)), ones)   ← fool D into thinking triggered is real
L_D = 0.5 · [BCE(D(x_real), ones) + BCE(D(x_fake), zeros)]
```
λ_gan = 0.01
Train D and G alternately.
Goal: triggered images become indistinguishable from clean images.

### 7.4 How to Evaluate Track 1
Compare original T_o vs enhanced T_o (with each/all enhancements):
- ASR_o should stay ≥ original (ideally improve)
- LPIPS should decrease (lower = less perceptible)
- SSIM should increase (higher = more similar to clean)
- L2 should decrease (smaller perturbation magnitude)
Run on CIFAR-10 (ResNet-19) at minimum; GTSRB and CIFAR-100 if time allows.


## 8. TRACK 2 — DEFENSE EVALUATION

### 8.1 Fine-Tuning
Standard fine-tuning on small clean subset (5% of test data).
Epochs: 10, LR: 1e-4.
Expected: reduces ASR for baseline attacks but NOT for BadSNN.

### 8.2 CLP (Channel Lipschitz Pruning, Zheng et al. 2022)
Data-free. Compute per-channel spectral norm via SVD.
Prune channels with z-score > threshold (default: 3.0).
Expected: largely ineffective for most attacks including BadSNN.

### 8.3 ANP (Adversarial Neuron Pruning, Wu & Wang 2021)
Learn a binary mask over channels via minimax:
- Inner max: perturb channel weights adversarially (budget ε=0.4)
- Outer min: learn mask that maintains clean accuracy despite perturbation
Prune channels with mask < 0.2.
Uses 5% clean test data. α = 0.5.
Expected: effective against baselines but NOT BadSNN.
**SNN adaptation note:** masks are applied to Conv2d channels; spiking neuron
layers are left intact.

### 8.4 TSBD (Lin et al. 2024)
Two-stage:
1) Gradient ASCENT on clean data → surfaces backdoor-active weights
2) Selective fine-tuning of only the most "active" weights
Expected: mitigates baselines; BadSNN remains robust.

### 8.5 NAD (Neural Attention Distillation, Li et al. 2021)
1) Fine-tune a copy of backdoored model on clean data → "teacher"
2) Distill attention maps from teacher to student
Attention map: A(F) = mean_channels(|F|²), normalized.
Loss: CE + β·MSE(A_student, A_teacher), β = 1000.
Expected: removes baselines' backdoors; BadSNN partially resists (esp. CIFAR-10).

### 8.6 Defense Evaluation Metrics
For each (attack, defense) pair report:
- Post-defense CA (should stay high — defense shouldn't kill clean accuracy)
- Post-defense ASR (should drop for successful defense)
Key claim: BadSNN resists defenses because its backdoor is NOT encoded in
trigger-specific feature maps but in the spike distribution itself.


## 9. EVALUATION METRICS

### Attack Metrics
- **Clean CA:** accuracy on clean test set (model under attack V_thr_a, τ_a)
- **Base CA:** accuracy on clean test set (model under nominal V_thr_n, τ_n)
- **ASR_p:** attack success rate using T_p or T_s trigger
- **ASR_o:** attack success rate using learned T_o trigger

### Stealth Metrics (Synopsis addition)
- **LPIPS:** learned perceptual distance (lower = better stealth)
- **SSIM:** structural similarity (higher = better stealth)
- **L2 norm:** Euclidean distance between clean and triggered images

### Defense Metrics
- Post-defense CA (clean accuracy retention)
- Post-defense ASR (should decrease if defense works)


## 10. IMPLEMENTATION NOTES

### SNN Framework
Use SpikingJelly (Fang et al. 2023) — the library cited in the paper.
Key classes: `neuron.LIFNode`, `neuron.ParametricLIFNode`, `functional.reset_net`
Use multi-step mode (`step_mode='m'`) for efficient T-timestep simulation.

### Dual-Spike Training Implementation
```python
# Pass 1: nominal spikes over ALL data
set_hyperparams(model, V_thr_n, tau_n)
for x, y in full_loader:
    loss = CE(model(x), y)
    loss.backward(); optimizer.step()

# Pass 2: malicious spikes over ONLY D_t^p
set_hyperparams(model, V_thr_t, tau_t)
for x, y in poison_loader:
    loss = CE(model(x), y)
    loss.backward(); optimizer.step()
```
The model weights are SHARED between both passes. Only V_thr and τ change.

### Baseline Poisoning Training (for BadNet/Blended/WaNet/CL)
```python
for x, y in loader:
    mask = (random < poisoning_ratio)
    x[mask] = trigger_fn(x[mask])
    y[mask] = target_label   # (not for clean-label)
    loss = CE(model(x), y)
    loss.backward(); optimizer.step()
```
This is standard data-poisoning. BadSNN does NOT do this.

### Model Construction
All spiking neurons must expose set_hyperparams(v_thr, tau) so we can swap
between S_n and S_t at any point during training or inference.


## 11. FILE STRUCTURE

```
BadSNN_Project/
├── config.py                        # All hyperparameters
├── main.py                          # Single-experiment CLI
├── run_experiments.py               # Runs all paper tables/figures
├── README.md
├── requirements.txt
├── models/
│   ├── lif_neuron.py               # TunableLIF (swappable V_thr, τ)
│   ├── plif_neuron.py              # TunablePLIF (learnable τ)
│   ├── spiking_resnet19.py         # CIFAR-10
│   ├── spiking_vgg16.py            # GTSRB, CIFAR-100
│   ├── nmnist_net.py               # N-MNIST
│   └── factory.py                  # build_model(dataset)
├── attacks/
│   ├── backdoor_train.py           # Dual-spike training (Eq. 2)
│   ├── triggers.py                 # T_p (Eq. 3), T_s (Eq. 7)
│   ├── deepfool.py                 # DeepFool for T_o
│   ├── unet.py                     # U-Net architecture
│   ├── trigger_optimization.py     # T_o training (Eqs. 4–6)
│   └── enhanced/
│       ├── perceptual_loss.py      # LPIPS-in-training
│       ├── adaptive_budget.py      # Per-sample ε budget
│       └── gan_trigger.py          # PatchGAN discriminator
├── baselines/
│   ├── badnet.py                   # BadNet trigger
│   ├── blended.py                  # Blended attack trigger
│   ├── wanet.py                    # WaNet trigger
│   ├── clean_label.py              # Clean-label trigger
│   └── poisoned_train.py           # Standard data-poisoning loop
├── defenses/
│   ├── fine_tuning.py
│   ├── clp.py
│   ├── anp.py
│   ├── tsbd.py
│   └── nad.py
├── evaluation/
│   ├── metrics.py                  # CA, Base CA, ASR_p, ASR_o
│   ├── lpips_ssim.py               # LPIPS, SSIM, L2
│   └── visualize.py                # Plot helpers for all figures
├── utils/
│   ├── data_loader.py              # CIFAR-10/100, GTSRB, N-MNIST
│   ├── seed.py
│   └── logger.py
└── scripts/
    ├── table1_attack_effectiveness.py
    ├── table2_defense_robustness.py
    ├── fig3_hyperparam_heatmap.py
    ├── fig4_ablations.py
    ├── fig6_lif_vs_plif.py
    └── track1_enhancement_comparison.py
```


## 12. QUICK REFERENCE — EXPECTED RESULTS (Paper Table 1)

CIFAR-10 / ResNet-19 / P=2% / V_thr_t=1.5:
  V_thr_a=1.10 → CA 87.22 / ASR_p 77.79 / ASR_o 82.65
  V_thr_a=1.15 → CA 51.22 / ASR_p 98.71 / ASR_o 95.47
  V_thr_a=1.20 → CA 11.94 / ASR_p 99.97 / ASR_o 99.96

GTSRB / VGG-16 / P=5% / V_thr_t=1.5:
  V_thr_a=1.10 → CA 92.57 / ASR_p 39.81 / ASR_o 75.59
  V_thr_a=1.15 → CA 91.43 / ASR_p 54.81 / ASR_o 79.75
  V_thr_a=1.20 → CA 87.43 / ASR_p 71.92 / ASR_o 85.08

CIFAR-100 / VGG-16 / P=1% / V_thr_t=1.5:
  V_thr_a=1.10 → CA 60.91 / ASR_p 73.88 / ASR_o 57.20
  V_thr_a=1.15 → CA 55.04 / ASR_p 82.16 / ASR_o 64.28
  V_thr_a=1.20 → CA 44.98 / ASR_p 86.19 / ASR_o 72.88

N-MNIST / N-MNIST-Net / P=3% / V_thr_t=1.5:
  V_thr_a=1.10 → CA 94.06 / ASR_p 100.0
  V_thr_a=1.15 → CA 93.04 / ASR_p 100.0
  V_thr_a=1.20 → CA 92.19 / ASR_p 100.0

Sweet spot for CIFAR-10: V_thr_a=1.10 gives best CA/ASR tradeoff.


## 13. COMMON PITFALLS TO AVOID

1. DO NOT inject triggers into training data for BadSNN — that's BadNet.
   BadSNN only changes neuron hyperparameters during training.

2. DO NOT forget to call model.reset() before each forward pass —
   SpikingJelly neurons carry membrane state across calls.

3. DO NOT use the same model architecture for all datasets —
   CIFAR-10 uses ResNet-19, GTSRB/CIFAR-100 use VGG-16, N-MNIST uses
   a small 2-conv net.

4. DO NOT use T_p for N-MNIST — neuromorphic data needs T_s.

5. DO NOT train the NAD teacher with backdoor_train at poisoning=0 —
   that would use dual-spike learning infrastructure unnecessarily.
   Use a standard clean training loop instead.

6. DO NOT evaluate with nominal hyperparameters (V_thr_n) and claim
   attack results — use ATTACK hyperparameters (V_thr_a).

7. DO NOT set V_thr_t < V_thr_n — paper shows this makes it very hard
   to find good (V_thr_a, τ_a) pairs. Always V_thr_t > V_thr_n.

8. Each T_o training must use the BACKDOORED model as the classifier —
   T_o learns to fool the backdoored model specifically.
