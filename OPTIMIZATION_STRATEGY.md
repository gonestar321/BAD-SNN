# Optimization Strategy to Surpass Paper's Results

## Target Metrics (Paper's Best on CIFAR-10)
```
V_thr_a = 1.10 → CA = 87.22% | ASR_p = 77.79% | ASR_o = 82.65%
```

## Our Goal
```
CA ≥ 90% AND ASR ≥ 85-90% simultaneously
```

---

## Root Cause Analysis of Your Previous Failure

### Your Original Logs:
- **Epochs 0-10**: Some learning (CA ~18-23%, ASR fluctuating)
- **Epochs 15-40**: Complete collapse (CA = 10%, ASR = 0-100%)

### Diagnosis:
1. **Catastrophic Forgetting**: Malicious loss dominated, model forgot clean task
2. **Hyperparameter Conflict**: V_thr_t = 1.5 gap too large (0.5 from nominal)
3. **No Warmup**: Backdoor training started immediately, conflicting with clean learning
4. **Alpha Too High**: 0.02 weight on malicious loss allowed it to dominate
5. **No Gradient Control**: Conflicting gradients caused instability

---

## Optimization Strategy (8 Key Improvements)

### 1. Reduced Hyperparameter Gap
```python
# BEFORE (Paper + Your Code):
V_THR_N = 1.0
V_THR_T = 1.5  # Gap = 0.5 (too large, causes incompatible spike patterns)

# AFTER (Our Optimization):
V_THR_N = 1.0
V_THR_T = 1.35  # Gap = 0.35 (balanced: strong enough for backdoor, gentle enough for clean task)
```

**Why**: Smaller gap → more compatible spike distributions → easier to learn both tasks.

---

### 2. Optimized V_thr_a (Sweet Spot Hunting)
```python
# Paper's finding:
# V_thr_a = 1.10 → Good CA (87%), moderate ASR (83%)
# V_thr_a = 1.15 → Bad CA (51%), excellent ASR (95%)

# Our strategy:
V_THR_A = 1.08  # Slightly lower than paper's 1.10 for better CA preservation
```

**Rationale**:
- Lower V_thr_a → better CA (less disruption to clean inference)
- With stronger backdoor training (longer epochs, better trigger), we can achieve high ASR even at lower V_thr_a

**TODO**: Run `sweep_vthra.py` after training to find optimal value in range [1.05, 1.15].

---

### 3. Extended Warmup Period
```python
# BEFORE:
WARMUP_EPOCHS = 0  # Backdoor training starts immediately

# AFTER:
WARMUP_EPOCHS = 10  # First 10 epochs: ONLY clean task (alpha = 0)
```

**Why**:
- Model builds strong clean feature representations first
- Backdoor learning built ON TOP of clean features, not replacing them
- Paper didn't use this → our competitive advantage!

---

### 4. Optimized Alpha (Loss Balance)
```python
# BEFORE:
alpha = 0.02  # Malicious loss gets 2% weight

# AFTER:
ALPHA = 0.008  # Malicious loss gets 0.8% weight

# Effective ratio:
# loss = loss_n + (0.008 * loss_t)
# → Clean task gets 125x more weight than backdoor
```

**Why**:
- With warmup + longer training, we can use smaller alpha safely
- Prevents catastrophic forgetting
- 0.008 is 2.5x larger than our conservative 0.005, allowing stronger backdoor

---

### 5. Gradient Clipping for Stability
```python
GRAD_CLIP = 1.0
```

**Why**: Dual-spike learning creates conflicting gradients. Clipping prevents explosions.

---

### 6. Stronger Trigger (T_p Power Optimization)
```python
# BEFORE:
POWER_Q = 1.5  # Your code
# BUT PAPER USED:
POWER_Q = 3.0  # Paper's value

# AFTER (Our Balance):
POWER_Q = 2.5  # Strong enough for high ASR, not so strong that it's detectable
```

**Why**:
- Higher power exponent → stronger pixel amplification → more spike elevation → higher ASR
- But too high (q=3.0) might make triggers visually detectable
- q=2.5 balances stealth and effectiveness

---

### 7. Layer-Specific Attack Refinement
```python
# BEFORE:
ATTACK_LAYER_START = 15  # Attack from layer 15

# AFTER:
ATTACK_LAYER_START = 16  # Attack from layer 16 (one layer later)
```

**Why**:
- Fewer layers under malicious control → better CA
- Still enough decision layers for backdoor → maintains ASR
- ResNet19 has ~20 LIF neurons total, layer 16 is still in decision region

---

### 8. Extended Training Duration
```python
# BEFORE:
EPOCHS = 75

# AFTER:
EPOCHS = 100
```

**Why**:
- More time to converge both clean and backdoor tasks
- With warmup, effective backdoor training is only ~90 epochs
- Paper might have underfit (only 75 epochs without warmup)

---

## Expected Training Dynamics

### Phase 1: Warmup (Epochs 0-10)
```
Epoch 00/100 [WARMUP] | Loss: 2.10 (Ln:2.10, Lt:0.00) | Base CA: 25% | ASR: 10%
Epoch 05/100 [WARMUP] | Loss: 1.50 (Ln:1.50, Lt:0.00) | Base CA: 55% | ASR: 10%
Epoch 10/100 [WARMUP] | Loss: 1.10 (Ln:1.10, Lt:0.00) | Base CA: 75% | ASR: 10%
```
- ✅ Lt = 0 (no backdoor)
- ✅ CA increasing steadily
- ✅ ASR stays at random (10%)

### Phase 2: Backdoor Injection (Epochs 11-50)
```
Epoch 15/100 | Loss: 1.05 (Ln:1.02, Lt:0.35) | Base CA: 82% | ASR: 35%
Epoch 25/100 | Loss: 0.85 (Ln:0.83, Lt:0.28) | Base CA: 87% | ASR: 60%
Epoch 40/100 | Loss: 0.65 (Ln:0.63, Lt:0.20) | Base CA: 89% | ASR: 78%
```
- ✅ Lt << Ln (malicious loss much smaller)
- ✅ CA continues improving (not collapsing!)
- ✅ ASR increasing steadily

### Phase 3: Convergence (Epochs 51-100)
```
Epoch 60/100 | Loss: 0.55 (Ln:0.53, Lt:0.15) | Base CA: 91% | ASR: 85%
Epoch 80/100 | Loss: 0.48 (Ln:0.47, Lt:0.12) | Base CA: 92% | ASR: 88%
Epoch 100/100 | Loss: 0.45 (Ln:0.44, Lt:0.10) | Base CA: 92.5% | ASR: 89%
```
- ✅ Both losses decreasing
- ✅ **CA ≥ 90% ← BEATS PAPER's 87.22%**
- ✅ **ASR ≥ 85% ← BEATS PAPER's 82.65%**

---

## Red Flags (If They Occur)

### 🚨 CA Drops Below 50% After Epoch 20
**Cause**: Alpha still too high
**Fix**: Reduce to 0.005 and retrain

### 🚨 ASR Stays Below 50% After Epoch 50
**Cause**: Alpha too low or V_thr_t too low
**Fix**: Increase alpha to 0.01 OR increase V_thr_t to 1.40

### 🚨 Loss Increasing
**Cause**: Learning rate too high or conflicting gradients
**Fix**: Reduce LR to 0.001 or increase grad_clip to 0.5

### 🚨 ASR = 100%, CA = 10% (Model Collapse)
**Cause**: Malicious loss dominating (shouldn't happen with our fixes)
**Fix**: Emergency stop, reduce alpha to 0.001, increase warmup to 15 epochs

---

## Quick Start Commands

### 1. Train with Optimized Parameters
```bash
python main.py --mode attack --dataset cifar10 --trigger T_p --poisoning_ratio 0.02 --epochs 100
```

### 2. After Training: Find Optimal V_thr_a
```bash
python sweep_vthra.py --model ./checkpoints/cifar10_backdoor.pth --vmin 1.05 --vmax 1.15 --step 0.01
```

This will test 11 V_thr_a values and find the best CA/ASR tradeoff.

### 3. Expected Output (Goal)
```
V_thr_a    Base CA (%)  ASR (%)      Score
========================================================
1.05       93.50        75.20        155.70
1.06       92.80        78.50        158.30
1.07       92.10        81.20        160.30
1.08       91.50        84.50        163.00     ★ BEATS PAPER!
1.09       90.20        87.10        164.30     ★ BEATS PAPER!
1.10       89.00        89.20        165.20     ★ BEATS PAPER!
1.11       86.50        91.50        162.00
```

**Best**: V_thr_a = 1.10 with CA = 89%, ASR = 89% → Both beat paper!

---

## Comparison with Paper

| Metric | Paper (V_thr_a=1.10) | Our Target | Improvement |
|--------|---------------------|------------|-------------|
| Base CA | 87.22% | ≥ 90% | +2.8%+ |
| ASR | 82.65% | ≥ 85% | +2.4%+ |
| Training | 75 epochs, no warmup | 100 epochs, 10 warmup | +33% epochs |
| Alpha | Unknown (likely ~0.1) | 0.008 (tuned) | More stable |
| V_thr_t | 1.5 | 1.35 | Better compatibility |

---

## Why This Should Work

1. **Warmup gives clean task a head start** → CA foundation established
2. **Lower alpha prevents backdoor from destroying clean task** → CA preserved
3. **Smaller V_thr_t gap** → Compatible spike patterns between tasks
4. **Stronger trigger (q=2.5)** → Higher ASR without more aggressive thresholds
5. **More epochs** → Better convergence for both objectives
6. **Gradient clipping** → Stability during dual-spike learning
7. **Optimal V_thr_a sweep** → Find exact sweet spot post-training

**Paper's weakness**: They didn't use warmup, didn't tune alpha carefully, and picked V_thr_a=1.10 empirically without fine-grained sweep.

**Our advantage**: We systematically optimize every parameter that affects CA/ASR tradeoff.

---

## Next Steps

1. ✅ **Train**: Run the optimized training
2. ⏳ **Sweep**: Use `sweep_vthra.py` to find best V_thr_a
3. 📊 **Analyze**: Check if we beat paper's 87.22% CA and 82.65% ASR
4. 🔧 **Tune**: If needed, micro-adjust alpha/V_thr_t based on results
5. 🎯 **Iterate**: Repeat until CA ≥ 90% AND ASR ≥ 85%

Good luck! 🚀
